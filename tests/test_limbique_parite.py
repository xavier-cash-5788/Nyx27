"""
Parité : l'ANCIEN code (main.py) et le NOUVEAU module (nyx.limbique) reçoivent les mêmes entrées et doivent
produire les mêmes sorties : classification, ton, lissage, dérive, apaisement du sommeil, effets du feedback,
prompt et lecture de l'analyse émotionnelle. (Ignorés si main.py est absent.)

Écarts volontaires, donc NON comparés : une valeur NaN/Infinity (ancien code : émotion extrême ; nouveau : repli
neutre) et un JSON dont les clés sont dans le désordre ou entouré de texte (ancien : lecture approximative).
"""
import importlib.util
import itertools
import os
import random

import pytest

from fakes import installer_faux_modules
from nyx.config import ConfigNyx
from nyx.limbique import EMOTIONS, Limbique, classer_emotion, instruction_ton

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def ancien(monkeypatch, tmp_path):
    chemin = os.path.join(RACINE, "main.py")
    if not os.path.exists(chemin):
        pytest.skip("main.py absent : parité ignorée")
    installer_faux_modules(monkeypatch)
    spec = importlib.util.spec_from_file_location("legacy_main", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.AgentCerebralNyx(chemin_db=str(tmp_path / "db"))
    agent._ecrire_dans_journal = lambda *a, **k: None
    return agent


def regler(ancien, etat):
    ancien.valence, ancien.activation, ancien.dominance = etat


def etat_de(ancien):
    return (ancien.valence, ancien.activation, ancien.dominance)


def test_emotion_dominante_identique_sur_une_grille_fine(ancien):
    V = [-1, -0.5, -0.3001, -0.3, -0.2999, 0, 0.2999, 0.3, 0.3001, 0.5, 1]
    A = [0, 0.3, 0.5999, 0.6, 0.6001, 0.8, 1]
    D = [0, 0.3, 0.5999, 0.6, 0.6001, 0.8, 1]
    for v, a, d in itertools.product(V, A, D):
        regler(ancien, (v, a, d))
        assert classer_emotion(v, a, d) == ancien._calculer_emotion_dominante(), (v, a, d)


def test_instruction_de_ton_identique_pour_chaque_emotion(ancien):
    for nom, (emoji, _, _) in EMOTIONS.items():
        assert instruction_ton(nom, emoji) == ancien._generer_instruction_emotionnelle_concise(nom, emoji), nom
    assert instruction_ton("Inconnue", "x") == ancien._generer_instruction_emotionnelle_concise("Inconnue", "x") == ""


def test_lissage_identique_sur_des_sequences_aleatoires(ancien):
    rnd = random.Random(1234)
    for _ in range(20):
        depart = (rnd.uniform(-1, 1), rnd.uniform(0, 1), rnd.uniform(0, 1))
        regler(ancien, depart)
        lb = Limbique(ConfigNyx.defauts(), etat=depart)
        for _ in range(100):
            stimulus = (rnd.uniform(-1, 1), rnd.uniform(0, 1), rnd.uniform(0, 1))
            ancien._mettre_a_jour_emotions(*stimulus)
            lb.mettre_a_jour(*stimulus)
            assert lb.etat().vad == pytest.approx(etat_de(ancien), abs=1e-12)


@pytest.mark.parametrize("dt", [0.1, 1, 60, 900, 5000, 10 ** 6, 0, -5])
def test_derive_identique(ancien, dt):
    depart = (0.7, 0.9, 0.2)
    regler(ancien, depart)
    lb = Limbique(ConfigNyx.defauts(), etat=depart)
    ancien._deriver_emotions(dt)
    lb.deriver(dt)
    assert lb.etat().vad == pytest.approx(etat_de(ancien), abs=1e-12)


def test_apaisement_du_sommeil_identique(ancien):
    """Le vrai `_consolider_memoire` de l'ancien code (LLM neutralisé) contre `apaiser()`.

    NB : l'ancien code fait un early-return quand l'hippocampe est vide (pas d'apaisement).
    Il faut donc RÉINITIALISER l'hippocampe à chaque itération pour que les deux codes
    atteignent l'apaisement à chaque tour.
    """
    ancien._appeler_llm = lambda *a, **k: {"message": {"content": '{"concepts": []}'}}
    meta = {
        "created_at": "2026-10-05 12:00:00", "dernier_acces": "2026-10-05 12:00:00",
        "force_base": 0.9, "reactivations": 0, "type": "souvenir_brut",
        "valence": 0.0, "activation": 0.5, "dominance": 0.5,
    }
    for depart in [(0.8, 0.9, 0.1), (-0.6, 0.2, 0.9), (0.0, 0.5, 0.5)]:
        # Reset de l'hippocampe avant chaque itération (sinon le souvenir est consommé)
        ids = ancien.hippocampe.get().get("ids", [])
        if ids:
            ancien.hippocampe.delete(ids=ids)
        ancien.hippocampe.add(documents=["un souvenir"], ids=["s1"], metadatas=[meta])

        regler(ancien, depart)
        lb = Limbique(ConfigNyx.defauts(), etat=depart)
        ancien._consolider_memoire()
        lb.apaiser()
        assert lb.etat().vad == pytest.approx(etat_de(ancien), abs=1e-12), depart


@pytest.mark.parametrize("depart", [(0.0, 0.5, 0.5), (0.95, 0.95, 0.5), (-0.95, 0.05, 0.5), (1.0, 1.0, 0.5),
                                    (-1.0, 0.0, 0.5), (0.3, 0.2, 0.7)])
@pytest.mark.parametrize("positif", [True, False])
def test_effet_du_feedback_identique(ancien, depart, positif):
    ancien.dernier_stimulus, ancien.derniere_reponse = "question", "réponse"
    regler(ancien, depart)
    lb = Limbique(ConfigNyx.defauts(), etat=depart)
    ancien.donner_feedback(1 if positif else -1)
    lb.retour_utilisateur(positif)
    assert lb.etat().vad == pytest.approx(etat_de(ancien), abs=1e-12)


def _faux_llm_ancien(ancien, contenu):
    appels = []

    def faux(messages, format=None, temperature=None, modele=None):
        appels.append((messages, format, temperature))
        return {"message": {"content": contenu}}
    ancien._appeler_llm = faux
    return appels


class FauxLLM:
    def __init__(self, contenu):
        self.contenu, self.appels = contenu, []

    def chat(self, messages, format=None, temperature=None, timeout=None, **kw):
        self.appels.append((messages, format, temperature))
        return {"message": {"content": self.contenu}}


def test_prompt_et_parametres_de_l_analyse_identiques(ancien):
    appels_a = _faux_llm_ancien(ancien, '{"valence": 0.1, "activation": 0.2, "dominance": 0.3}')
    llm = FauxLLM('{"valence": 0.1, "activation": 0.2, "dominance": 0.3}')
    for stimulus in ("Bonjour", 'Un "message" avec des guillemets', "ligne 1\nligne 2", "émojis 😀 et accents é"):
        ancien._analyser_stimulus_emotionnel(stimulus)
        Limbique(ConfigNyx.defauts(), llm=llm).analyser(stimulus)
    assert llm.appels == appels_a and len(appels_a) == 4


@pytest.mark.parametrize("contenu", [
    '{"valence": 0.5, "activation": 0.4, "dominance": 0.3}',
    '{"valence": 2, "activation": -1, "dominance": 5}',
    '{"valence": "0.5", "activation": "0.4", "dominance": "0.3"}',
    "valence 0.4, activation 0.8, dominance 0.2",
    "0.1 0.2 0.3 0.4",
    "-0.5, 0.9, 0.1",
    '{"valence": 0.5, "activation": 0.4}',
    '{"valence": 0.5}', "rien du tout", "", "0.1 0.2",
])
def test_lecture_de_l_analyse_identique(ancien, contenu):
    _faux_llm_ancien(ancien, contenu)
    attendu = ancien._analyser_stimulus_emotionnel("x")
    obtenu = Limbique(ConfigNyx.defauts(), llm=FauxLLM(contenu)).analyser("x")
    assert obtenu.vad == pytest.approx(attendu)


def test_une_exception_du_modele_donne_le_meme_repli_neutre(ancien):
    def panne(*a, **k):
        raise RuntimeError("panne")
    ancien._appeler_llm = panne

    class LLMEnPanne:
        def chat(self, *a, **k):
            raise RuntimeError("panne")
    assert ancien._analyser_stimulus_emotionnel("x") == Limbique(ConfigNyx.defauts(), llm=LLMEnPanne()).analyser("x").vad

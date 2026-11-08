import math
import random
import threading

import pytest

from nyx.config import ConfigNyx
from nyx.limbique import (EMOJIS, EMOTIONS, NEUTRE, AnalyseEmotion, Humeur, Limbique, Perception, classer_emotion,
                          construire_prompt_analyse, instruction_ton, lire_vad)
from nyx.llm import ClientLLM, ErreurLLM, LLMAnnule, LLMTimeout


class FauxLLM:
    """Client de test : renvoie un contenu fixé (ou lève une erreur) et enregistre les appels."""

    def __init__(self, contenu=None, erreur=None):
        self.contenu, self.erreur, self.appels = contenu, erreur, []

    def chat(self, messages, **kw):
        self.appels.append((messages, kw))
        if self.erreur:
            raise self.erreur
        return {"message": {"role": "assistant", "content": self.contenu}}


@pytest.fixture
def cfg():
    return ConfigNyx.defauts()


def limbique(cfg, contenu='{"valence": 0.6, "activation": 0.7, "dominance": 0.5}', erreur=None, **kw):
    return Limbique(cfg, llm=FauxLLM(contenu, erreur), **kw)


# ================================================================ classification et ton

def test_les_onze_emotions_sont_atteignables():
    atteintes = {classer_emotion(v, a, d)[0]
                 for v in (-1, -0.5, 0, 0.5, 1) for a in (0, 0.5, 1) for d in (0, 0.5, 1)}
    assert atteintes == set(EMOTIONS) and len(atteintes) == 11


@pytest.mark.parametrize("vad, attendu", [
    ((0.31, 0.61, 0.61), "Excitation joyeuse"), ((0.31, 0.61, 0.6), "Surprise agréable"),
    ((0.31, 0.6, 0.61), "Sérénité confiante"), ((0.31, 0.6, 0.6), "Contentement calme"),
    ((0.3, 0.61, 0.61), "Curiosité intense"), ((0.0, 0.61, 0.6), "Intérêt modéré"),
    ((-0.3, 0.61, 0.61), "Colère"), ((-0.31, 0.61, 0.61), "Colère"), ((-0.31, 0.61, 0.6), "Peur"),
    ((-0.31, 0.6, 0.61), "Frustration"), ((-0.31, 0.6, 0.6), "Tristesse"),
    ((-0.3, 0.6, 0.9), "Frustration"), ((0.3, 0.6, 0.9), "Neutralité"), ((0.0, 0.5, 0.5), "Neutralité"),
])
def test_frontieres_de_classification(vad, attendu):
    nom, emoji = classer_emotion(*vad)
    assert nom == attendu and emoji == EMOJIS[attendu]
    # les seuils sont asymétriques (v = -0.3 est « négatif », v = +0.3 est « central ») : c'est le comportement
    # de l'ancien code, verrouillé par le test de parité.


def test_instruction_ton():
    assert instruction_ton("Excitation joyeuse") == "Ton actuel : enthousiaste 🎉. Réponds avec énergie."
    assert instruction_ton("Neutralité", "X") == "Ton actuel : neutre X. Sois factuel."
    assert instruction_ton("Inconnue") == ""
    assert all(instruction_ton(nom) for nom in EMOTIONS)


def test_instance_etat_instruction_emotion(cfg):
    lb = Limbique(cfg, etat=(0.8, 0.9, 0.9))
    assert lb.emotion_dominante() == ("Excitation joyeuse", "🎉")
    assert lb.instruction_ton() == "Ton actuel : enthousiaste 🎉. Réponds avec énergie."
    assert lb.emotion_dominante(Humeur(-0.9, 0.9, 0.2)) == ("Peur", "😨")


# ================================================================ lecture des réponses du modèle

@pytest.mark.parametrize("brut, attendu, source", [
    ('{"valence": 0.5, "activation": 0.4, "dominance": 0.3}', (0.5, 0.4, 0.3), "json"),
    ('```json\n{"valence": -0.2, "activation": 0.9, "dominance": 0.1}\n```', (-0.2, 0.9, 0.1), "json"),
    ('Analyse : {"dominance": 0.8, "valence": 0.1, "activation": 0.2} voilà', (0.1, 0.2, 0.8), "json"),
    ('{"valence": "0.5", "activation": "0.4", "dominance": "0.3"}', (0.5, 0.4, 0.3), "json"),
    ('{"valence": 7, "activation": -3, "dominance": 9}', (1.0, 0.0, 1.0), "json"),
    ("valence 0.4, activation 0.8, dominance 0.2", (0.4, 0.8, 0.2), "nombres"),
    ("-0.5 0.9 0.1 0.4", (-0.5, 0.9, 0.1), "nombres"),
    ('{"valence": 0.5, "activation": 0.4}\n0.9 0.8', (0.5, 0.4, 0.9), "nombres"),
])
def test_lire_vad_valide(brut, attendu, source):
    *vad, src = lire_vad(brut)
    assert tuple(vad) == pytest.approx(attendu) and src == source


@pytest.mark.parametrize("brut", ["", "   ", "rien du tout", '{"valence": 0.5}', "0.1 0.2", None, 42, [],
                                  '{"valence": NaN, "activation": 0.5, "dominance": 0.5}',
                                  '{"valence": Infinity, "activation": 0.5, "dominance": 0.5}'])
def test_lire_vad_refuse_l_inexploitable_et_les_valeurs_non_finies(brut):
    with pytest.raises(ValueError):
        lire_vad(brut)


def test_le_prompt_contient_le_message_et_les_reperes():
    p = construire_prompt_analyse('Je suis "content" !')
    assert 'Message : "Je suis "content" !"' in p and "Repères d'échelle" in p and '"dominance" (0 à 1)' in p


# ================================================================ analyse

def test_analyser_appelle_le_modele_avec_les_bons_parametres(cfg):
    lb = limbique(cfg)
    r = lb.analyser("Bonjour Nyx")
    assert isinstance(r, AnalyseEmotion) and r.ok and r.source == "json" and r.erreur is None
    assert r.vad == pytest.approx((0.6, 0.7, 0.5))
    (messages, kw), = lb.llm.appels
    assert messages == [{"role": "user", "content": construire_prompt_analyse("Bonjour Nyx")}]
    assert kw == {"format": "json", "temperature": 0, "timeout": 30.0}


@pytest.mark.parametrize("contenu, erreur", [
    (None, ErreurLLM("modèle indisponible")), (None, LLMTimeout("trop long")), (None, RuntimeError("bug")),
    ("réponse sans chiffres", None), ("", None), ('{"valence": NaN, "activation": 1, "dominance": 1}', None),
])
def test_analyser_se_replie_sur_le_neutre_sans_lever(cfg, contenu, erreur):
    r = limbique(cfg, contenu, erreur).analyser("x")
    assert r.vad == NEUTRE and r.ok is False and r.source == "defaut" and r.erreur


def test_analyser_laisse_passer_l_annulation(cfg):
    with pytest.raises(LLMAnnule):
        limbique(cfg, erreur=LLMAnnule("annulé")).analyser("x")


def test_analyser_sans_client_llm(cfg):
    with pytest.raises(RuntimeError):
        Limbique(cfg).analyser("x")


def test_percevoir_analyse_puis_met_a_jour(cfg):
    lb = limbique(cfg, '{"valence": 1.0, "activation": 1.0, "dominance": 1.0}')
    p = lb.percevoir("génial !")
    assert isinstance(p, Perception) and p.avant == Humeur(*NEUTRE) and p.analyse.ok
    assert p.apres == lb.etat() and p.apres.valence > 0.2 and p.apres.activation > p.avant.activation
    assert (p.emotion, p.emoji) == lb.emotion_dominante(p.apres)


def test_percevoir_avec_analyse_echouee_tire_l_humeur_vers_le_neutre(cfg):
    lb = limbique(cfg, erreur=ErreurLLM("panne"), etat=(0.8, 0.9, 0.9))
    p = lb.percevoir("x")
    assert p.analyse.ok is False and 0 < lb.valence < 0.8, "comme avant : un message « neutre » apaise l'humeur"


def test_avec_le_vrai_client_llm(cfg):
    """Le Limbique accepte le vrai ClientLLM (mêmes paramètres) : format JSON, température 0, délai court."""
    class Backend:
        def __init__(self):
            self.appels = []

        def chat(self, **kw):
            self.appels.append(kw)
            yield {"message": {"role": "assistant", "content": '{"valence": 0.2, "activation": 0.6, "dominance": 0.4}'}}

    backend = Backend()
    lb = Limbique(cfg, llm=ClientLLM(cfg, backend=backend))
    r = lb.analyser("test")
    assert r.ok and r.vad == pytest.approx((0.2, 0.6, 0.4))
    kw = backend.appels[0]
    assert kw["format"] == "json" and kw["options"]["temperature"] == 0 and kw["model"] == cfg.model_name


# ================================================================ dynamique de l'humeur

def test_mettre_a_jour_suit_la_formule(cfg):
    lb = Limbique(cfg, etat=(0.2, 0.4, 0.6))
    lb.mettre_a_jour(0.8, 1.0, 0.0)
    f, k = 0.3, 0.95
    assert lb.valence == pytest.approx(((1 - f) * 0.2 + f * 0.8) * k)
    assert lb.activation == pytest.approx(((1 - f) * 0.4 + f * 1.0) * k + 0.5 * 0.05)
    assert lb.dominance == pytest.approx(((1 - f) * 0.6 + f * 0.0) * k + 0.5 * 0.05)


def test_le_neutre_est_un_point_fixe_et_les_bornes_tiennent(cfg):
    lb = Limbique(cfg)
    for _ in range(50):
        lb.mettre_a_jour(*NEUTRE)
    assert lb.etat().vad == pytest.approx(NEUTRE)
    rnd = random.Random(7)
    for _ in range(2000):
        lb.mettre_a_jour(rnd.choice((-1.0, 1.0)), rnd.choice((0.0, 1.0)), rnd.choice((0.0, 1.0)))
        assert -1 <= lb.valence <= 1 and 0 <= lb.activation <= 1 and 0 <= lb.dominance <= 1


def test_deriver_vers_le_neutre(cfg):
    lb = Limbique(cfg, etat=(0.8, 0.9, 0.1))
    lb.deriver(cfg.TAU_EMOTION)                      # une constante de temps : facteur e^-1
    k = math.exp(-1)
    assert lb.etat().vad == pytest.approx((0.8 * k, 0.5 + 0.4 * k, 0.5 - 0.4 * k))
    for dt in (0, -5):
        avant = lb.etat()
        lb.deriver(dt)
        assert lb.etat() == avant, "dt <= 0 : aucun effet"
    lb.deriver(10 ** 6)
    assert lb.etat().vad == pytest.approx(NEUTRE, abs=1e-9)


def test_deriver_relit_tau_dans_la_config(cfg):
    a, b = Limbique(cfg, etat=(1.0, 0.5, 0.5)), Limbique(cfg, etat=(1.0, 0.5, 0.5))
    a.deriver(100)
    cfg.definir("TAU_EMOTION", 100)
    b.deriver(100)
    assert b.valence < a.valence and b.valence == pytest.approx(math.exp(-1))


def test_apaiser_et_retour_utilisateur(cfg):
    lb = Limbique(cfg, etat=(0.8, 0.9, 0.1))
    lb.apaiser()
    assert lb.etat().vad == pytest.approx((0.4, 0.7, 0.3))
    lb.apaiser(0.0)
    assert lb.etat().vad == pytest.approx(NEUTRE)
    lb.retour_utilisateur(True)
    assert lb.etat().vad == pytest.approx((0.2, 0.6, 0.5))
    lb.retour_utilisateur(False)
    lb.retour_utilisateur(False)
    assert lb.etat().vad == pytest.approx((-0.2, 0.4, 0.5))
    for _ in range(20):
        lb.retour_utilisateur(False)
    assert lb.valence == -1.0 and lb.activation == 0.0, "plafonné"
    for _ in range(20):
        lb.retour_utilisateur(True)
    assert lb.valence == 1.0 and lb.activation == 1.0


def test_etat_humeur_et_reinitialiser(cfg):
    lb = Limbique(cfg, etat=(0.5, 0.7, 0.3))
    assert lb.etat() == Humeur(0.5, 0.7, 0.3) and lb.humeur() == (0.5, 0.7)
    with pytest.raises(Exception):
        lb.etat().valence = 1.0                       # l'instantané est figé
    lb.reinitialiser()
    assert lb.etat().vad == NEUTRE


def test_acces_concurrents(cfg):
    lb = Limbique(cfg)
    erreurs = []

    def agir(n):
        try:
            for i in range(300):
                lb.mettre_a_jour(1.0 if (i + n) % 2 else -1.0, 1.0, 0.0)
                lb.deriver(1.0)
                lb.retour_utilisateur(i % 2 == 0)
                lb.etat()
        except Exception as e:  # noqa: BLE001
            erreurs.append(e)
    threads = [threading.Thread(target=agir, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not erreurs and -1 <= lb.valence <= 1 and 0 <= lb.activation <= 1

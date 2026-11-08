"""
Parité : l'ANCIEN code (main.py) et le NOUVEAU module (nyx.consolidation) reçoivent les mêmes entrées et doivent
produire les mêmes sorties : oubli des faibles, consolidation des neutres en concepts, apaisement émotionnel,
écriture du journal de réveil. (Ignorés si main.py est absent.)
Écarts volontaires, donc NON comparés :
- l'ancien `dormir()` fait plus de choses (reset des pulsions, etc.) que le nouveau `Consolidation.dormir()` ;
- l'ancien code appelle `_ecrire_dans_journal` sur `self`, le nouveau reçoit un `Journal` injecté.
"""
import importlib.util
import os
import uuid

import pytest

from fakes import FausseCollection, installer_faux_modules
from nyx.config import ConfigNyx
from nyx.consolidation import Consolidation
from nyx.journal import Journal
from nyx.limbique import Limbique
from nyx.memoire import Memoire

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FauxLLM:
    """Client de test : renvoie un contenu fixé et enregistre les appels."""
    def __init__(self, contenu=None):
        self.contenu = contenu
        self.appels = []

    def chat(self, messages, **kw):
        self.appels.append((messages, kw))
        return {"message": {"role": "assistant", "content": self.contenu}}


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


@pytest.fixture
def nouveau(ancien):
    cfg = ConfigNyx.defauts()
    memoire = Memoire(cfg, ancien.hippocampe, None)
    journal = Journal(cfg, ancien.journal_intime)
    limbique = Limbique(cfg)
    return Consolidation(cfg, memoire, journal, limbique)


def _vider(col):
    ids = col.get().get("ids", [])
    if ids:
        col.delete(ids=ids)


def ajouter_souvenir(col, texte, age_h, force, v=0.0, a=0.5, typ="souvenir_brut"):
    """Ajoute un souvenir avec un dernier accès à `age_h` heures."""
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(hours=age_h)).strftime("%Y-%m-%d %H:%M:%S")
    col.add(
        documents=[texte],
        metadatas=[{
            "created_at": ts,
            "dernier_acces": ts,
            "force_base": force,
            "reactivations": 0,
            "type": typ,
            "valence": v,
            "activation": a,
            "dominance": 0.5,
        }],
        ids=[f"id-{uuid.uuid4()}"],
    )


# ------------------------------------------------------------------ oubli des faibles
def test_oubli_des_faibles_identique(ancien, nouveau):
    """Les souvenirs faibles doivent être oubliés de la même manière."""
    ajouter_souvenir(ancien.hippocampe, "faible neutre", age_h=100, force=0.5)
    ajouter_souvenir(ancien.hippocampe, "faible marquant", age_h=100, force=0.5, v=-0.9, a=0.9)
    ajouter_souvenir(ancien.hippocampe, "concept recent", age_h=5, force=0.85, typ="concept_consolidé")
    ajouter_souvenir(ancien.hippocampe, "neutre recent", age_h=1, force=0.6)

    ancien._consolider_memoire()
    ancien_restants = set(ancien.hippocampe.get()['documents'])

    nouveau.dormir()
    nouveau_restants = set(nouveau.memoire.col.get()['documents'])

    assert "faible neutre" not in ancien_restants
    assert "faible neutre" not in nouveau_restants
    assert "faible marquant" in ancien_restants
    assert "faible marquant" in nouveau_restants
    assert "concept recent" in ancien_restants
    assert "concept recent" in nouveau_restants
    assert "neutre recent" in ancien_restants
    assert "neutre recent" in nouveau_restants


# ------------------------------------------------------------------ consolidation des neutres
# Les concepts DOIVENT être valides pour _concept_valide (pas de "concept ", pas de markdown).
CONCEPTS_VALIDES = [
    "La synthèse des souvenirs neutres forme un récit cohérent.",
    "Une autre synthèse relie les apprentissages à l'identité.",
]


def test_consolidation_des_neutres_identique(ancien, nouveau):
    """Les souvenirs neutres doivent être consolidés en concepts de la même manière."""
    import json
    contenu = json.dumps({"concepts": CONCEPTS_VALIDES})

    # --- Ancien code ---
    for i in range(4):
        ajouter_souvenir(ancien.hippocampe, f"souvenir neutre {i}", age_h=1, force=0.6)
    ancien._appeler_llm = lambda *a, **k: {"message": {"content": contenu}}

    ancien._consolider_memoire()
    data = ancien.hippocampe.get()
    ancien_concepts = [d for d, m in zip(data['documents'], data['metadatas'])
                       if m and m.get('type') == 'concept_consolidé']

    # --- Nouveau code : repartir d'un état propre (les deux partagent la même collection) ---
    _vider(ancien.hippocampe)
    for i in range(4):
        ajouter_souvenir(ancien.hippocampe, f"souvenir neutre {i}", age_h=1, force=0.6)
    nouveau.llm = FauxLLM(contenu)

    nouveau.dormir()
    nouveau_concepts = [s.texte for s in nouveau.memoire.tous() if s.type == 'concept_consolidé']

    assert len(ancien_concepts) == 2
    assert len(nouveau_concepts) == 2
    assert set(ancien_concepts) == set(nouveau_concepts)


def test_consolidation_sans_llm(ancien, nouveau):
    """Sans LLM, les souvenirs neutres doivent être conservés."""
    for i in range(4):
        ajouter_souvenir(ancien.hippocampe, f"souvenir neutre {i}", age_h=1, force=0.6)

    ancien._appeler_llm = None
    ancien._consolider_memoire()
    ancien_restants = set(ancien.hippocampe.get()['documents'])

    nouveau.llm = None
    nouveau.dormir()
    nouveau_restants = set(s.texte for s in nouveau.memoire.tous())

    assert len(ancien_restants) == 4
    assert len(nouveau_restants) == 4


# ------------------------------------------------------------------ apaisement émotionnel
def test_apaisement_emotionnel_identique(ancien, nouveau):
    """L'apaisement émotionnel doit être identique.

    NB : `_consolider_memoire` (ancien) et `dormir` (nouveau) font tous les deux un early-return
    quand l'hippocampe est vide. Il FAUT donc au moins un souvenir pour que l'apaisement ait lieu.
    """
    # 1 souvenir récent (non oubliable) pour que les deux codes atteignent l'apaisement
    ajouter_souvenir(ancien.hippocampe, "souvenir frais", age_h=0, force=0.6)

    ancien.valence, ancien.activation, ancien.dominance = 0.8, 0.9, 0.1
    nouveau.limbique.valence = 0.8
    nouveau.limbique.activation = 0.9
    nouveau.limbique.dominance = 0.1

    ancien._consolider_memoire()
    ancien_vad = (ancien.valence, ancien.activation, ancien.dominance)

    nouveau.dormir()
    nouveau_vad = nouveau.limbique.etat().vad

    assert nouveau_vad == pytest.approx(ancien_vad, abs=1e-6)
    assert abs(nouveau_vad[0] - 0.4) < 1e-6  # 0.8 * 0.5
    assert abs(nouveau_vad[1] - 0.7) < 1e-6  # 0.9 * 0.5 + 0.5 * 0.5
    assert abs(nouveau_vad[2] - 0.3) < 1e-6  # 0.1 * 0.5 + 0.5 * 0.5


# ------------------------------------------------------------------ journal de réveil
def test_journal_de_reveil_identique(ancien, nouveau):
    """Le journal de réveil doit être écrit de la même manière."""
    contenu_ancien = "Ancien : phase de sommeil terminée."
    contenu_nouveau = "Nouveau : phase de sommeil terminée."

    # Rétablir la vraie `_ecrire_dans_journal` (la fixture l'a mockée)
    ancien.__dict__.pop("_ecrire_dans_journal", None)
    ancien._appeler_llm = lambda *a, **k: {"message": {"content": contenu_ancien}}

    class FauxLLM:
        def __init__(self, c):
            self.contenu = c
        def chat(self, messages, **kw):
            return {"message": {"content": self.contenu}}
    nouveau.journal.llm = FauxLLM(contenu_nouveau)

    ajouter_souvenir(ancien.hippocampe, "faible neutre", age_h=100, force=0.5)
    ajouter_souvenir(ancien.hippocampe, "neutre recent", age_h=1, force=0.6)

    ancien_avant = ancien._dernieres_reflexions(1)
    ancien._consolider_memoire()
    ancien_apres = ancien._dernieres_reflexions(1)
    assert len(ancien_apres) > len(ancien_avant)

    # Les deux codes partagent la même collection : vider avant le nouveau
    _vider(ancien.journal_intime)

    nouveau_avant = nouveau.journal.dernieres_reflexions(1)
    nouveau.dormir()
    nouveau_apres = nouveau.journal.dernieres_reflexions(1)
    assert len(nouveau_apres) > len(nouveau_avant)


# ------------------------------------------------------------------ cas limites
def test_dormir_memoire_vide(ancien, nouveau):
    """Dormir avec une mémoire vide ne doit pas planter."""
    ancien._consolider_memoire()
    nouveau.dormir()
    assert ancien.hippocampe.count() == 0
    assert nouveau.memoire.compter() == 0


def test_dormir_sans_souvenirs_neutres(ancien, nouveau):
    """Dormir sans souvenirs neutres ne doit pas appeler le LLM."""
    ajouter_souvenir(ancien.hippocampe, "marquant", age_h=100, force=0.5, v=-0.9, a=0.9)
    ajouter_souvenir(ancien.hippocampe, "concept", age_h=5, force=0.85, typ="concept_consolidé")

    ancien._appeler_llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ne doit pas être appelé"))
    nouveau.llm = FauxLLM()

    ancien._consolider_memoire()
    nouveau.dormir()

    assert len(nouveau.llm.appels) == 0

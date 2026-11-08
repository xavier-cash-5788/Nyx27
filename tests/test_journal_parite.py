"""
Parité : l'ANCIEN code (main.py) et le NOUVEAU module (nyx.journal) reçoivent les mêmes entrées et doivent
produire les mêmes sorties : tri des entrées, dernières réflexions, identité récente, enregistrement
(anti-doublon), écriture via LLM. (Ignorés si main.py est absent.)

Écarts volontaires, donc NON comparés :
- l'ancien code appelle `_calculer_emotion_dominante()` sur `self`, le nouveau reçoit l'émotion en paramètre ;
- l'ancien code utilise `time.time()` directement, le nouveau reçoit une fonction `maintenant` injectable ;
- l'ancien `_catchup_identitaire()` ne retourne rien (effet de bord), le nouveau retourne la liste lue.
"""
import importlib.util
import os
import time

import pytest

from fakes import installer_faux_modules
from nyx.config import ConfigNyx
from nyx.journal import Journal

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FauxLLM:
    """Client de test : renvoie un contenu fixé et enregistre les appels."""
    def __init__(self, contenu=None, erreur=None):
        self.contenu, self.erreur, self.appels = contenu, erreur, []

    def chat(self, messages, **kw):
        self.appels.append((messages, kw))
        if self.erreur:
            raise self.erreur
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
    agent._ecrire_dans_journal = lambda *a, **k: None  # neutralisé pour les tests de lecture
    return agent


@pytest.fixture
def nouveau(ancien):
    cfg = ConfigNyx.defauts()
    return Journal(cfg, ancien.journal_intime)


def ajouter_entree(col, texte, epoch, type_evenement="test", emotion="Neutralité",
                   valence=0.0, activation=0.5, dominance=0.5):
    """Ajoute une entrée au journal avec un epoch fixé."""
    col.add(
        documents=[texte],
        metadatas=[{
            "timestamp": "2026-10-05 12:00:00",
            "epoch": epoch,
            "type_evenement": type_evenement,
            "emotion": emotion,
            "valence": valence,
            "activation": activation,
            "dominance": dominance,
        }],
        ids=[f"id-{epoch}"]
    )


def _retablir_ecriture_journal(ancien, contenu):
    """Rétablit la vraie `_ecrire_dans_journal` de l'ancien code (la fixture l'a mockée)
    et branche un faux LLM qui renvoie `contenu`."""
    ancien.__dict__.pop("_ecrire_dans_journal", None)  # retrouve la méthode de classe
    ancien._appeler_llm = lambda messages, **kw: {"message": {"content": contenu}}


def _vider_journal(ancien):
    """Vide la collection du journal pour repartir d'un état propre."""
    ids = ancien.journal_intime.get().get("ids", [])
    if ids:
        ancien.journal_intime.delete(ids=ids)


# ------------------------------------------------------------------ tri et lecture
def test_epoch_de_identique(ancien, nouveau):
    """`_epoch_de` (ancien) et `_epoch_de` (nouveau) doivent donner le même résultat."""
    cas = [
        {"epoch": 1000.0, "timestamp": "2026-10-05 12:00:00"},
        {"timestamp": "2026-10-05 12:00:00"},  # pas d'epoch : parsé depuis timestamp
        {"timestamp": "pas une date"},          # invalide : 0.0
        {},                                     # vide : 0.0
        None,                                   # None : 0.0
    ]
    for meta in cas:
        assert nouveau._epoch_de(meta) == ancien._epoch_de(meta), meta


def test_entrees_triees_identiques(ancien, nouveau):
    """Les entrées triées par epoch doivent être identiques."""
    base = time.time()
    ajouter_entree(ancien.journal_intime, "réflexion C", base + 3)
    ajouter_entree(ancien.journal_intime, "réflexion A", base + 1)
    ajouter_entree(ancien.journal_intime, "réflexion B", base + 2)

    ancien_entrees = [(e, d) for e, d, _ in ancien._entrees_journal_triees()]
    nouveau_entrees = [(e.epoch, e.texte) for e in nouveau.entrees_triees()]

    assert ancien_entrees == nouveau_entrees


def test_dernieres_reflexions_identiques(ancien, nouveau):
    """Les n dernières réflexions doivent être identiques."""
    base = time.time()
    for i, texte in enumerate(["A", "B", "C", "D", "E"]):
        ajouter_entree(ancien.journal_intime, texte, base + i)

    for n in (1, 2, 3, 5, 10):
        assert nouveau.dernieres_reflexions(n) == ancien._dernieres_reflexions(n)


def test_charger_identite_recente_identique(ancien, nouveau):
    """L'identité récente doit être construite de la même manière."""
    base = time.time()
    for i, texte in enumerate(["réflexion A", "réflexion B", "réflexion C"]):
        ajouter_entree(ancien.journal_intime, texte, base + i)

    ancien._charger_identite_recente(2)
    nouveau.charger_identite_recente(2)

    assert nouveau.identite_recente == ancien.identite_recente
    assert "réflexion B" in nouveau.identite_recente
    assert "réflexion C" in nouveau.identite_recente
    assert "réflexion A" not in nouveau.identite_recente


def test_catchup_identitaire_identique(ancien, nouveau):
    """Le catchup identitaire doit lire les mêmes réflexions.

    L'ancien ne retourne rien (effet de bord uniquement) ; le nouveau retourne la liste.
    On compare donc les EFFETS : l'identité récente mise à jour.
    """
    base = time.time()
    for i, texte in enumerate(["réflexion A", "réflexion B", "réflexion C", "réflexion D", "réflexion E"]):
        ajouter_entree(ancien.journal_intime, texte, base + i)

    ancien._catchup_identitaire()
    nouveau.catchup_identitaire()

    # Effet identique : `identite_recente` contient les 3 dernières réflexions
    assert nouveau.identite_recente == ancien.identite_recente
    assert "réflexion C" in nouveau.identite_recente
    assert "réflexion D" in nouveau.identite_recente
    assert "réflexion E" in nouveau.identite_recente
    assert "réflexion A" not in nouveau.identite_recente


# ------------------------------------------------------------------ enregistrement
def test_enregistrer_identique(ancien, nouveau):
    """L'enregistrement d'une réflexion doit produire le même résultat.

    Les deux codes écrivent dans la même collection (celle de l'ancien) : on la vide entre
    les deux appels pour partir d'un état identique et éviter la détection de doublon.
    """
    ancien.valence, ancien.activation, ancien.dominance = 0.5, 0.7, 0.6

    # Ancien code
    ancien_reflexion = ancien._enregistrer_journal("Test de parité", "test")
    assert ancien_reflexion == "Test de parité"
    assert ancien.journal_intime.count() == 1

    # Repartir d'un état propre pour le nouveau
    _vider_journal(ancien)
    assert ancien.journal_intime.count() == 0

    # Nouveau code (même état émotionnel)
    nouveau_reflexion = nouveau.enregistrer(
        "Test de parité", "test",
        emotion="Neutralité", valence=0.5, activation=0.7, dominance=0.6
    )

    assert nouveau_reflexion == ancien_reflexion
    assert ancien.journal_intime.count() == 1


def test_anti_doublon_identique(ancien, nouveau):
    """L'anti-doublon doit fonctionner de la même manière."""
    # Ajouter une première entrée
    ajouter_entree(ancien.journal_intime, "Réflexion initiale", time.time())

    # Tenter d'ajouter une réflexion très proche
    ancien.valence, ancien.activation, ancien.dominance = 0.0, 0.5, 0.5
    ancien_reflexion = ancien._enregistrer_journal("Réflexion initiale", "test")

    nouveau_reflexion = nouveau.enregistrer(
        "Réflexion initiale", "test",
        emotion="Neutralité", valence=0.0, activation=0.5, dominance=0.5
    )

    # Les deux doivent ignorer le doublon
    assert ancien_reflexion is None
    assert nouveau_reflexion is None
    assert ancien.journal_intime.count() == 1  # seule la première entrée


# ------------------------------------------------------------------ écriture via LLM
def test_ecrire_via_llm_identique(ancien, nouveau):
    """L'écriture via LLM doit appeler le modèle avec les mêmes paramètres.

    La fixture `ancien` a mocké `_ecrire_dans_journal` : on la rétablit et on branche un faux LLM
    pour comparer le résultat réel des deux codes.
    """
    contenu = "Ceci est une réflexion de test générée par le LLM."
    faux_llm = FauxLLM(contenu)

    # Ancien : rétablir la vraie écriture + brancher le faux LLM
    _retablir_ecriture_journal(ancien, contenu)
    ancien.valence, ancien.activation, ancien.dominance = 0.3, 0.6, 0.5
    ancien_reflexion = ancien._ecrire_dans_journal("Contexte de test", "test")
    assert ancien_reflexion == contenu
    assert ancien.journal_intime.count() == 1

    # Repartir d'un état propre pour le nouveau (sinon anti-doublon)
    _vider_journal(ancien)
    assert ancien.journal_intime.count() == 0

    # Nouveau code
    nouveau.llm = faux_llm
    nouveau_reflexion = nouveau.ecrire(
        "Contexte de test", "test",
        emotion="Neutralité", emoji="😐", vad=(0.3, 0.6, 0.5)
    )

    assert nouveau_reflexion == ancien_reflexion
    assert ancien.journal_intime.count() == 1
    assert len(faux_llm.appels) == 1

    # Vérifier que le prompt contient les mêmes informations
    messages, kw = faux_llm.appels[0]
    prompt_user = messages[1]["content"]
    assert "Contexte de test" in prompt_user
    assert "test" in prompt_user
    assert "Neutralité" in prompt_user


def test_ecrire_via_llm_erreur(ancien, nouveau):
    """Une erreur du LLM doit être gérée de la même manière."""
    # Ancien : rétablir la vraie écriture + brancher un LLM en panne
    ancien.__dict__.pop("_ecrire_dans_journal", None)

    def panne(*a, **k):
        raise RuntimeError("panne")
    ancien._appeler_llm = panne

    ancien_reflexion = ancien._ecrire_dans_journal("Contexte", "test")
    assert ancien_reflexion is None
    assert ancien.journal_intime.count() == 0

    # Nouveau : même erreur
    nouveau.llm = FauxLLM(erreur=RuntimeError("panne"))
    nouveau_reflexion = nouveau.ecrire("Contexte", "test")

    assert nouveau_reflexion is None
    assert ancien.journal_intime.count() == 0


def test_ecrire_via_llm_sans_llm(ancien, nouveau):
    """Sans LLM, le nouveau code doit retourner None (l'ancien lève une exception)."""
    nouveau.llm = None
    assert nouveau.ecrire("Contexte", "test") is None

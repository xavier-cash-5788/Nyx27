"""
Parité : l'ANCIEN code (main.py) et le NOUVEAU module (nyx.pulsions) reçoivent les mêmes entrées et doivent
produire les mêmes sorties : calcul de la fatigue, de l'envie de réfléchir, tick de l'horloge interne,
auto-reflexion. (Ignorés si main.py est absent.)
Écarts volontaires, donc NON comparés :
- l'ancien code a `calculer_pulsions` qui lit `self.debut_eveil`, `self.derniere_interaction`, etc. ;
  le nouveau `Pulsions` a les mêmes attributs, donc les formules sont identiques ;
- l'ancien `tick` appelle `self.dormir()` directement, le nouveau `Pulsions` appelle un callback `dormir` injecté ;
- l'ancien produit lui-même la pensée via `_appeler_llm`, le nouveau délègue à un callback `penser` injecté.
"""
import importlib.util
import os
import time
import uuid

import pytest

from fakes import FausseCollection, installer_faux_modules
from nyx.config import ConfigNyx
from nyx.limbique import Limbique
from nyx.memoire import Memoire
from nyx.pulsions import Pulsions

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


@pytest.fixture
def nouveau(ancien):
    cfg = ConfigNyx.defauts()
    memoire = Memoire(cfg, ancien.hippocampe, None)
    limbique = Limbique(cfg)
    return Pulsions(cfg, memoire, limbique)


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


# ------------------------------------------------------------------ calcul des pulsions
def test_calculer_pulsions_identique(ancien, nouveau):
    """Le calcul de la fatigue et de l'envie de réfléchir doit être identique."""
    t0 = time.time()

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = t0
    ancien.charge_msgs = 0.0
    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = t0
    nouveau.charge_msgs = 0.0

    for i in range(3):
        ajouter_souvenir(ancien.hippocampe, f"souvenir {i}", age_h=14, force=0.8)

    ancien_p = ancien.calculer_pulsions(t0)
    nouveau_p = nouveau.calculer(t0)

    assert nouveau_p["fatigue"] == pytest.approx(ancien_p["fatigue"], abs=1e-6)
    assert nouveau_p["envie_reflexion"] == pytest.approx(ancien_p["envie_reflexion"], abs=1e-6)
    assert nouveau_p["nb_declin"] == ancien_p["nb_declin"]
    assert nouveau_p["nb_faibles"] == ancien_p["nb_faibles"]
    assert nouveau_p["total"] == ancien_p["total"]

    ancien_p2 = ancien.calculer_pulsions(t0 + 3 * 3600)
    nouveau_p2 = nouveau.calculer(t0 + 3 * 3600)

    assert nouveau_p2["fatigue"] == pytest.approx(ancien_p2["fatigue"], abs=1e-6)
    assert nouveau_p2["fatigue"] > ancien_p["fatigue"]


def test_fatigue_avec_messages(ancien, nouveau):
    """Les messages doivent fatiguer de la même manière."""
    t0 = time.time()
    ancien.debut_eveil = ancien.derniere_interaction = t0
    ancien.charge_msgs = 0.5
    nouveau.debut_eveil = nouveau.derniere_interaction = t0
    nouveau.charge_msgs = 0.5

    ancien_p = ancien.calculer_pulsions(t0)
    nouveau_p = nouveau.calculer(t0)

    assert nouveau_p["fatigue"] == pytest.approx(ancien_p["fatigue"], abs=1e-6)
    assert abs(nouveau_p["fatigue"] - 0.5) < 0.02


def test_fatigue_avec_encombrement(ancien, nouveau):
    """L'encombrement de souvenirs presque oubliés doit fatiguer de la même manière."""
    t0 = time.time()
    ancien.debut_eveil = ancien.derniere_interaction = t0
    ancien.charge_msgs = 0.0
    nouveau.debut_eveil = nouveau.derniere_interaction = t0
    nouveau.charge_msgs = 0.0

    for i in range(4):
        ajouter_souvenir(ancien.hippocampe, f"souvenir faible {i}", age_h=100, force=0.5)

    ancien_p = ancien.calculer_pulsions(t0)
    nouveau_p = nouveau.calculer(t0)

    assert nouveau_p["fatigue"] == pytest.approx(ancien_p["fatigue"], abs=1e-6)
    assert nouveau_p["fatigue"] >= 0.25


# ------------------------------------------------------------------ tick
def test_tick_envie_dormir_identique(ancien, nouveau):
    """Le tick doit annoncer l'envie de dormir au même moment."""
    t0 = time.time()

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = ancien.dernier_tick = t0
    ancien.charge_msgs = 0.65
    ancien.envie_dormir_annoncee = False
    ancien.reflexion_auto_actif = False
    ancien.sommeil_auto_actif = True

    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = nouveau.dernier_tick = t0
    nouveau.charge_msgs = 0.65
    nouveau.envie_dormir_annoncee = False
    nouveau.cfg.definir("reflexion_auto_actif", False)
    nouveau.cfg.definir("sommeil_auto_actif", True)

    ancien_evenements = []
    nouveau_evenements = []
    ancien.sur_evenement = lambda genre, texte: ancien_evenements.append(genre)
    nouveau.sur_evenement = lambda genre, texte: nouveau_evenements.append(genre)

    ancien_resultat = ancien.tick(t0)
    nouveau_resultat = nouveau.tick(t0)

    assert ancien_resultat == "envie_dormir"
    assert nouveau_resultat == "envie_dormir"
    assert ancien_evenements == ["envie_dormir"]
    assert nouveau_evenements == ["envie_dormir"]


def test_tick_sommeil_auto_identique(ancien, nouveau):
    """Le sommeil automatique doit se déclencher au même moment."""
    t0 = time.time()

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = ancien.dernier_tick = t0
    ancien.charge_msgs = 0.95
    ancien.envie_dormir_annoncee = True
    ancien.sommeil_auto_actif = True
    ancien.reflexion_auto_actif = False

    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = nouveau.dernier_tick = t0
    nouveau.charge_msgs = 0.95
    nouveau.envie_dormir_annoncee = True
    nouveau.cfg.definir("sommeil_auto_actif", True)
    nouveau.cfg.definir("reflexion_auto_actif", False)

    ancien_evenements = []
    nouveau_evenements = []
    ancien.sur_evenement = lambda genre, texte: ancien_evenements.append(genre)
    nouveau.sur_evenement = lambda genre, texte: nouveau_evenements.append(genre)

    ancien.dormir = lambda: setattr(ancien, '_dormi', True)
    nouveau._dormir = lambda: setattr(nouveau, '_dormi', True)

    ancien_resultat = ancien.tick(t0 + 60)
    nouveau_resultat = nouveau.tick(t0 + 60)

    assert ancien_resultat == "dormir"
    assert nouveau_resultat == "dormir"
    assert ancien_evenements[-2:] == ["sommeil", "reveil"]
    assert nouveau_evenements[-2:] == ["sommeil", "reveil"]


def test_tick_pas_sommeil_en_conversation(ancien, nouveau):
    """Nyx ne doit pas s'endormir en pleine conversation."""
    t0 = time.time()

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = ancien.dernier_tick = t0
    ancien.charge_msgs = 0.95
    ancien.envie_dormir_annoncee = True
    ancien.sommeil_auto_actif = True

    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = nouveau.dernier_tick = t0
    nouveau.charge_msgs = 0.95
    nouveau.envie_dormir_annoncee = True
    nouveau.cfg.definir("sommeil_auto_actif", True)

    ancien.derniere_interaction = t0 + 10
    nouveau.derniere_interaction = t0 + 10

    ancien_resultat = ancien.tick(t0 + 10)
    nouveau_resultat = nouveau.tick(t0 + 10)

    assert ancien_resultat is None
    assert nouveau_resultat is None


# ------------------------------------------------------------------ auto-reflexion
def test_auto_reflexion_identique(ancien, nouveau):
    """L'auto-reflexion doit renforcer les mêmes souvenirs.

    NB : la production de la pensée est déléguée au callback `penser` du nouveau module
    (elle n'est pas faite par le module `Pulsions` lui-même).
    """
    t0 = time.time()

    for i in range(3):
        ajouter_souvenir(ancien.hippocampe, f"souvenir {i}", age_h=14, force=0.8)

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = t0
    ancien.charge_msgs = 0.0
    ancien.envie_dormir_annoncee = False
    ancien.sommeil_auto_actif = True
    ancien.reflexion_auto_actif = True

    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = t0
    nouveau.charge_msgs = 0.0
    nouveau.envie_dormir_annoncee = False
    nouveau.cfg.definir("sommeil_auto_actif", True)
    nouveau.cfg.definir("reflexion_auto_actif", True)

    contenu = "Je repense à ces souvenirs qui s'estompent."
    ancien._appeler_llm = lambda *a, **k: {"message": {"content": contenu}}
    nouveau._penser = lambda cibles: contenu

    ancien_evenements = []
    nouveau_evenements = []
    ancien.sur_evenement = lambda genre, texte: ancien_evenements.append(genre)
    nouveau.sur_evenement = lambda genre, texte: nouveau_evenements.append(genre)

    ancien_pensee = ancien.auto_reflexion(force=True, maintenant=t0)
    nouveau_pensee = nouveau.auto_reflexion(force=True, maintenant=t0)

    assert ancien_pensee == contenu
    assert nouveau_pensee == contenu
    assert ancien_evenements == ["auto_reflexion"]
    assert nouveau_evenements == ["auto_reflexion"]

    ancien_metas = ancien.hippocampe.get()['metadatas']
    nouveau_metas = nouveau.memoire.col.get()['metadatas']

    ancien_renforces = sum(1 for m in ancien_metas if int(m.get('reactivations', 0)) >= 1)
    nouveau_renforces = sum(1 for m in nouveau_metas if int(m.get('reactivations', 0)) >= 1)

    assert ancien_renforces >= 2
    assert nouveau_renforces >= 2


def test_auto_reflexion_anti_boucle(ancien, nouveau):
    """L'anti-boucle (DELAI_MIN_REFLEXION) doit empêcher deux pensées trop rapprochées via `tick`.

    NB IMPORTANT : `auto_reflexion(force=True)` est un ordre explicite — ni l'ancien ni le nouveau
    code n'y appliquent l'anti-boucle. C'est `tick` qui respecte DELAI_MIN_REFLEXION.
    """
    t0 = time.time()

    for i in range(3):
        ajouter_souvenir(ancien.hippocampe, f"souvenir {i}", age_h=14, force=0.8)

    contenu = "Je repense à ces souvenirs qui s'estompent et je veux les garder vivants."

    ancien._appeler_llm = lambda *a, **k: {"message": {"content": contenu}}
    nouveau._penser = lambda cibles: contenu

    ancien.debut_eveil = ancien.derniere_interaction = ancien.derniere_auto_reflexion = ancien.dernier_tick = t0
    ancien.charge_msgs = 0.0
    ancien.envie_dormir_annoncee = False
    ancien.sommeil_auto_actif = False
    ancien.reflexion_auto_actif = True
    ancien.ENVIE_PAR_MINUTE = 100.0

    nouveau.debut_eveil = nouveau.derniere_interaction = nouveau.derniere_auto_reflexion = nouveau.dernier_tick = t0
    nouveau.charge_msgs = 0.0
    nouveau.envie_dormir_annoncee = False
    nouveau.cfg.definir("sommeil_auto_actif", False)
    nouveau.cfg.definir("reflexion_auto_actif", True)
    nouveau.cfg.definir("ENVIE_PAR_MINUTE", 100.0)

    r1_a = ancien.tick(t0 + 15 * 60)
    r1_n = nouveau.tick(t0 + 15 * 60)
    assert r1_a == "auto_reflexion", f"ancien : attendu auto_reflexion, obtenu {r1_a}"
    assert r1_n == "auto_reflexion", f"nouveau : attendu auto_reflexion, obtenu {r1_n}"

    r2_a = ancien.tick(t0 + 15 * 60 + 10)
    r2_n = nouveau.tick(t0 + 15 * 60 + 10)
    assert r2_a is None, "ancien : deux pensées trop rapprochées devraient être bloquées"
    assert r2_n is None, "nouveau : deux pensées trop rapprochées devraient être bloquées"
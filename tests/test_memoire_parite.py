"""
Parité : l'ANCIEN code (main.py) et le NOUVEAU module (nyx.memoire) font tourner les mêmes données et doivent
donner les mêmes résultats. C'est le filet de sécurité de la refonte : si ces tests passent, la migration
n'a changé aucun comportement de la mémoire. (Ignorés si main.py est absent.)
"""
import importlib.util
import os
from datetime import datetime, timedelta

import pytest

from fakes import FausseCollection, embedder_distances, installer_faux_modules
from nyx.config import ConfigNyx
from nyx.memoire import FORMAT_TS, Memoire

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISTANCES = {"m1": 0.50, "m2": 0.55, "m3": 0.90, "m4": 0.70, "m5": 0.60, "m6": 1.00, "m7": 1.50, "m8": 0.65, "m9": 0.80}
HUMEURS = [(0.0, 0.5), (-0.9, 0.9), (0.7, 0.3), (-0.3, 0.8), (1.0, 1.0), (-1.0, 0.1)]


@pytest.fixture
def legacy(monkeypatch, tmp_path):
    chemin = os.path.join(RACINE, "main.py")
    if not os.path.exists(chemin):
        pytest.skip("main.py absent : parité ignorée")
    installer_faux_modules(monkeypatch)
    spec = importlib.util.spec_from_file_location("legacy_main", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.AgentCerebralNyx(chemin_db=str(tmp_path / "db"))
    agent.hippocampe = FausseCollection("legacy", embedder_distances(DISTANCES))
    return agent


def il_y_a(h):
    return (datetime.now() - timedelta(hours=h)).strftime(FORMAT_TS)


def remplir(col):
    def ajouter(doc, age, force, v=0.0, a=0.5, typ="souvenir_brut"):
        col.add(documents=[doc], ids=[doc], metadatas=[{
            "created_at": il_y_a(age), "dernier_acces": il_y_a(age), "force_base": force, "reactivations": 2,
            "type": typ, "valence": v, "activation": a, "dominance": 0.5}])
    ajouter("m1", 1, 0.6, 0.8, 0.7)
    ajouter("m2", 30, 0.8, -0.8, 0.7)
    ajouter("m3", 3, 0.6)
    ajouter("m4", 20, 0.85, typ="concept_consolidé")
    ajouter("m5", 120, 0.5, -0.95, 0.95)
    col.add(documents=["m6"], ids=["m6"], metadatas=[{"timestamp": il_y_a(5), "force_synaptique": 0.7}])   # ancien schéma
    ajouter("m7", 2, 0.9, -0.9, 0.9)
    ajouter("m8", 90, 0.5)
    col.inserer_brut("m9", "m9", None)


@pytest.fixture
def paire(legacy):
    remplir(legacy.hippocampe)
    return legacy, Memoire(ConfigNyx.defauts(), legacy.hippocampe, None)


def test_normaliser_identique(paire):
    ancien, nouveau = paire
    for meta in (None, {}, {"timestamp": "2026-01-01 10:00:00", "force_synaptique": 0.7},
                 {"created_at": "a", "dernier_acces": "b", "force_base": "0.4", "reactivations": "3", "type": "x",
                  "valence": "-0.2", "activation": 0.1, "dominance": 0.9}):
        a, n = ancien._normaliser_meta(meta), nouveau.normaliser(meta)
        if not (meta and ("timestamp" in meta or "created_at" in meta)):   # horodatage par défaut = « maintenant »
            a.pop("created_at"), a.pop("dernier_acces"), n.pop("created_at"), n.pop("dernier_acces")
        assert a == n


def test_force_et_statut_identiques_pour_chaque_souvenir(paire):
    ancien, nouveau = paire
    d = ancien.hippocampe.get()
    for doc_id, meta in zip(d["ids"], d["metadatas"]):
        assert nouveau.force_effective(meta) == pytest.approx(ancien._force_effective(meta), abs=1e-4), doc_id
        assert nouveau.est_marquant(meta) == ancien._est_marquant(meta), doc_id
        assert nouveau.est_protege(meta) == ancien._est_protege(meta), doc_id


@pytest.mark.parametrize("humeur", HUMEURS)
@pytest.mark.parametrize("n", [1, 3, 6])
def test_rappel_identique_pour_toutes_les_humeurs(paire, humeur, n):
    ancien, nouveau = paire
    ancien.valence, ancien.activation = humeur
    ids_a, docs_a, _ = ancien._rappeler_souvenirs("Q", n=n, humeur=True)
    nouveaux = nouveau.rappeler("Q", n=n, humeur=humeur)
    assert [s.id for s in nouveaux] == ids_a and [s.texte for s in nouveaux] == docs_a
    assert len(ids_a) > 0


def test_rappel_sans_humeur_identique(paire):
    ancien, nouveau = paire
    ancien.valence, ancien.activation = -0.9, 0.9
    ids_a, _, _ = ancien._rappeler_souvenirs("Q", n=4, humeur=False)
    assert [s.id for s in nouveau.rappeler("Q", n=4, humeur=None)] == ids_a


def test_rappel_identique_avec_des_reglages_differents(paire):
    ancien, nouveau = paire
    for nom, valeur in (("POIDS_HUMEUR", 0.9), ("SEUIL_DISTANCE", 0.75), ("CANDIDATS_RAPPEL", 4)):
        setattr(ancien, nom, valeur)
        nouveau.cfg.definir(nom, valeur)
    for humeur in HUMEURS:
        ancien.valence, ancien.activation = humeur
        ids_a, _, _ = ancien._rappeler_souvenirs("Q", n=3, humeur=True)
        assert [s.id for s in nouveau.rappeler("Q", n=3, humeur=humeur)] == ids_a, humeur


def test_en_declin_identique(paire):
    ancien, nouveau = paire
    declin_a, faibles_a, total_a = ancien._souvenirs_en_declin()
    declin_n, faibles_n, total_n = nouveau.en_declin()
    assert (faibles_n, total_n) == (faibles_a, total_a)
    assert [s.id for s in declin_n] == [i for _, i, _ in declin_a]
    for s, (force, _, _) in zip(declin_n, declin_a):
        assert s.force == pytest.approx(force, abs=1e-4)


def test_renforcement_identique(paire):
    ancien, nouveau = paire
    copie_ancien, copie_nouveau = ancien.hippocampe.cloner(), ancien.hippocampe.cloner()
    ancien.hippocampe = copie_ancien
    nouveau.col = copie_nouveau
    ids = ["m1", "m2", "m5", "m6", "m9", "inexistant"]       # inclut ancien schéma, sans métadonnées, inconnu
    ancien._renforcer_souvenirs(ids)
    nouveau.renforcer(ids)
    for doc_id in copie_ancien.rows:
        _, ma = copie_ancien.rows[doc_id]
        _, mn = copie_nouveau.rows[doc_id]
        if ma is None:
            assert mn is None
            continue
        assert set(ma) == set(mn), doc_id
        assert mn["force_base"] == pytest.approx(ma["force_base"], abs=1e-4), doc_id
        assert mn["reactivations"] == ma["reactivations"], doc_id


def test_validation_des_concepts_identique(paire):
    ancien, _ = paire
    cas = ["Voici la consolidation", "Bien sûr, je suis ravi", "**Concept 1 : x** assez long pour passer ici",
           "Je fusionne les concepts suivants", "Concept 1 : La relation humaine", "x" * 14, "x" * 15, "x" * 250,
           "x" * 251, "ligne 1\nligne 2 assez longue", "- puce assez longue pour passer", "* puce longue assez ici",
           "• puce longue assez ici aussi", "# titre assez long pour passer", "Un vrai concept bien formé et valide.",
           "  espaces autour d'un concept valide  ", 42, None, ["liste"]]
    for c in cas:
        assert Memoire.concept_valide(c) == ancien._concept_valide(c), repr(c)


def test_le_tri_du_sommeil_oublie_exactement_les_memes_souvenirs(paire):
    """Le vrai `_consolider_memoire` de l'ancien code (LLM et journal neutralisés) contre `trier_pour_sommeil`."""
    ancien, nouveau = paire
    copie = ancien.hippocampe.cloner()
    ancien.hippocampe = copie
    ancien._appeler_llm = lambda *a, **k: {"message": {"content": '{"concepts": []}'}}
    ancien._ecrire_dans_journal = lambda *a, **k: None
    avant = set(copie.rows)
    ancien._consolider_memoire()
    oublies_ancien = avant - set(copie.rows)

    plan = nouveau.trier_pour_sommeil()
    assert {s.id for s in plan.oubli} == oublies_ancien
    assert oublies_ancien >= {"m8", "m9"}, "le jeu de données doit exercer l'oubli (faible + sans métadonnées)"
    assert "m5" in {s.id for s in plan.marquants} and "m5" not in oublies_ancien, "marquant très affaibli préservé"
    assert {s.id for s in plan.concepts} == {"m4"}

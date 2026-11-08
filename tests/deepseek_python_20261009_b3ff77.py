import os

import pytest

from nyx.documents import DocumentStore, InfoDocument, normaliser_nom


def test_normaliser_nom():
    assert normaliser_nom("Notes de Philo") == "notes_de_philo"
    assert normaliser_nom("Réflexion sur l'IA") == "reflexion_sur_l_ia"
    assert normaliser_nom("   espaces   ") == "espaces"
    assert normaliser_nom("") == ""
    assert normaliser_nom(None) == ""
    assert len(normaliser_nom("x" * 200)) <= 60


@pytest.fixture
def store(tmp_path):
    return DocumentStore(str(tmp_path / "docs"))


def test_creer_lire_listing(store):
    chemin = store.creer("ma note", "Contenu de test.")
    assert os.path.isfile(chemin)
    assert store.lire("ma note") == "Contenu de test."
    assert store.lire("ma_note") == "Contenu de test."
    assert store.existe("ma note")
    assert not store.existe("inconnu")
    docs = store.lister()
    assert len(docs) == 1 and isinstance(docs[0], InfoDocument)
    assert docs[0].nom == "ma_note" and docs[0].taille > 0


def test_creer_refuse_ecrasement(store):
    store.creer("x", "premier")
    with pytest.raises(FileExistsError):
        store.creer("x", "deuxième")
    assert store.lire("x") == "premier"
    store.creer("x", "deuxième", ecraser=True)
    assert store.lire("x") == "deuxième"


def test_ajouter_complete(store):
    store.creer("journal", "Ligne 1")
    store.ajouter("journal", "Ligne 2")
    assert store.lire("journal") == "Ligne 1\n\nLigne 2"
    store.ajouter("journal", "Ligne 3")
    assert store.lire("journal") == "Ligne 1\n\nLigne 2\n\nLigne 3"


def test_ajouter_cree_si_absent(store):
    store.ajouter("nouveau", "Première ligne")
    assert store.lire("nouveau") == "Première ligne"


def test_supprimer(store):
    store.creer("temporaire", "x")
    assert store.supprimer("temporaire")
    assert not store.supprimer("temporaire")
    assert store.lire("temporaire") is None


def test_chercher_dans_nom_et_contenu(store):
    store.creer("recette_pates", "Faire cuire les pâtes 10 minutes.")
    store.creer("notes_diverses", "Rien à voir ici.")
    store.creer("philo", "Réflexion sur la cuisine italienne et les pâtes fraîches.")
    resultats = store.chercher("pâtes")
    noms = {r.nom for r in resultats}
    assert "recette_pates" in noms and "philo" in noms
    assert "notes_diverses" not in noms


def test_chercher_requete_vide(store):
    store.creer("x", "y")
    assert store.chercher("") == []
    assert store.chercher("a") == []  # mots de moins de 3 caractères ignorés


def test_lister_trie_par_date_modif(store):
    import time
    store.creer("vieux", "1")
    time.sleep(0.05)
    store.creer("recent", "2")
    docs = store.lister()
    assert docs[0].nom == "recent"
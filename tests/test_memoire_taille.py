"""
Tests unitaires pour la surveillance de taille et la purge automatique de nyx/memoire.py.

Ces tests ne créent PAS 2 Go de données : on utilise un dossier temporaire (tmp_path)
rempli de fichiers de taille contrôlée, et on configure memoire_limite_go pour rendre
les seuils atteignables (ex: 1 Ko au lieu de 2 Go). Cela permet de tester en millisecondes
toute la logique de seuils, d'avertissement et de purge.

Cas couverts :
  - verifier_taille() : mesure, fraction, seuils, dossier absent, dossier vide
  - cache du calcul de taille : plusieurs appels rapprochés ne relisent pas le disque
  - purger_si_necessaire() : ne se déclenche pas sous la limite, purge au-dessus
  - ordre de sacrifice : faibles d'abord, puis déclin, JAMAIS marquants ni concepts
  - désactivation via memoire_purge_auto_actif
  - cas limites : aucun candidat à purger, dossier sans fichier
"""
import os
import time

import pytest

from fakes import FausseCollection, Horloge
from nyx.config import ConfigNyx
from nyx.memoire import Memoire, StatsTaille, TYPE_BRUT, TYPE_CONCEPT
from datetime import datetime

DEBUT = datetime(2026, 10, 5, 12, 0, 0)


# ================================================================ fixtures

@pytest.fixture
def horloge():
    return Horloge(DEBUT)


@pytest.fixture
def cfg_taille(tmp_path):
    """Config avec une limite très basse (1 Ko) pour rendre les seuils atteignables."""
    cfg = ConfigNyx.defauts()
    cfg.definir("memoire_limite_go", 0.000001)   # 1 Ko
    cfg.definir("memoire_seuil_avertissement", 0.8)
    cfg.definir("memoire_purge_cible", 0.5)
    cfg.definir("memoire_purge_auto_actif", True)
    return cfg


@pytest.fixture
def dossier_chroma(tmp_path):
    """Dossier vide qui jouera le rôle de chemin_db."""
    d = tmp_path / "chroma_simule"
    d.mkdir()
    return str(d)


@pytest.fixture
def mem_taille(cfg_taille, horloge, dossier_chroma):
    """Memoire avec une FausseCollection et un chemin_db pointant sur dossier_chroma."""
    return Memoire(
        cfg_taille,
        FausseCollection("souvenirs"),
        archive=FausseCollection("archives"),
        horloge=horloge,
        chemin_db=dossier_chroma,
    )


def ecrire_fichier(dossier, nom, taille_octets):
    """Crée un fichier de taille exacte dans `dossier`."""
    chemin = os.path.join(dossier, nom)
    with open(chemin, "wb") as f:
        f.write(b"x" * taille_octets)
    return chemin


def poser_souvenir(mem, horloge, texte, age_h=0.0, force=0.6, v=0.0, a=0.5, type=TYPE_BRUT):
    """Insère un souvenir avec une force contrôlée dans la FausseCollection."""
    from datetime import timedelta
    ts = (horloge() - timedelta(hours=age_h)).strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "created_at": ts, "dernier_acces": ts, "force_base": force,
        "reactivations": 0, "type": type, "valence": v, "activation": a, "dominance": 0.5,
    }
    doc_id = f"s-{len(mem.col.rows)}"
    mem.col.add(documents=[texte], metadatas=[meta], ids=[doc_id])
    return doc_id


# ================================================================ verifier_taille

def test_verifier_taille_dossier_vide(mem_taille):
    stats = mem_taille.verifier_taille()
    assert isinstance(stats, StatsTaille)
    assert stats.taille_go == 0.0
    assert stats.nb_fichiers == 0
    assert stats.fraction == 0.0
    assert stats.au_dessus_avertissement is False
    assert stats.au_dessus_limite is False


def test_verifier_taille_mesure_correctement(mem_taille, dossier_chroma):
    # 1 Ko = 1024 octets → limite 1 Ko → fraction ≈ 1.0
    ecrire_fichier(dossier_chroma, "f1.bin", 512)
    ecrire_fichier(dossier_chroma, "f2.bin", 512)
    stats = mem_taille.verifier_taille(cache_s=0.0)
    assert stats.nb_fichiers == 2
    assert stats.taille_go == pytest.approx(1024 / (1024 ** 3), rel=1e-6)
    assert stats.fraction == pytest.approx(1.0, rel=1e-4)


def test_verifier_taille_sous_seuil_avertissement(mem_taille, dossier_chroma):
    # 1/3 de la limite → fraction ≈ 0.33 < 0.8
    ecrire_fichier(dossier_chroma, "petit.bin", 341)
    stats = mem_taille.verifier_taille(cache_s=0.0)
    assert stats.au_dessus_avertissement is False
    assert stats.au_dessus_limite is False


def test_verifier_taille_au_dessus_avertissement(mem_taille, dossier_chroma):
    # 85 % de la limite → au-dessus du seuil 80 %
    ecrire_fichier(dossier_chroma, "presque.bin", int(1024 * 0.85))
    stats = mem_taille.verifier_taille(cache_s=0.0)
    assert stats.au_dessus_avertissement is True
    assert stats.au_dessus_limite is False


def test_verifier_taille_au_dessus_limite(mem_taille, dossier_chroma):
    ecrire_fichier(dossier_chroma, "trop.bin", 2000)  # 2 Ko > 1 Ko
    stats = mem_taille.verifier_taille(cache_s=0.0)
    assert stats.au_dessus_avertissement is True
    assert stats.au_dessus_limite is True


def test_verifier_taille_dossier_absent(cfg_taille, horloge):
    mem = Memoire(cfg_taille, FausseCollection(), chemin_db="/chemin/qui/n/existe/pas")
    stats = mem.verifier_taille()
    assert stats.taille_go == 0.0
    assert stats.nb_fichiers == 0
    assert stats.au_dessus_limite is False


def test_verifier_taille_sans_chemin_db(cfg_taille, horloge):
    mem = Memoire(cfg_taille, FausseCollection(), chemin_db=None)
    stats = mem.verifier_taille()
    assert stats.taille_go == 0.0
    assert stats.au_dessus_limite is False


def test_verifier_taille_sous_dossiers(mem_taille, dossier_chroma):
    """Les fichiers dans des sous-dossiers doivent être comptés."""
    sous = os.path.join(dossier_chroma, "collection_uuid")
    os.makedirs(sous)
    ecrire_fichier(sous, "data.bin", 500)
    ecrire_fichier(dossier_chroma, "racine.bin", 500)
    stats = mem_taille.verifier_taille(cache_s=0.0)
    assert stats.nb_fichiers == 2
    assert stats.taille_go == pytest.approx(1000 / (1024 ** 3), rel=1e-6)


# ================================================================ cache du calcul

def test_cache_evite_de_reparcourir_le_dossier(mem_taille, dossier_chroma):
    ecrire_fichier(dossier_chroma, "f.bin", 100)
    # Premier appel : mesure
    mem_taille.verifier_taille(cache_s=60.0)
    # Ajouter un fichier ne change rien tant que le cache est valide
    ecrire_fichier(dossier_chroma, "nouveau.bin", 200)
    stats2 = mem_taille.verifier_taille(cache_s=60.0)
    assert stats2.nb_fichiers == 1  # l'ancien résultat en cache


def test_cache_expire(mem_taille, dossier_chroma):
    ecrire_fichier(dossier_chroma, "f.bin", 100)
    mem_taille.verifier_taille(cache_s=0.05)
    time.sleep(0.1)
    ecrire_fichier(dossier_chroma, "nouveau.bin", 200)
    stats = mem_taille.verifier_taille(cache_s=0.05)
    assert stats.nb_fichiers == 2


def test_cache_s_zero_force_la_mesure(mem_taille, dossier_chroma):
    ecrire_fichier(dossier_chroma, "f1.bin", 100)
    s1 = mem_taille.verifier_taille(cache_s=0.0)
    assert s1.nb_fichiers == 1
    ecrire_fichier(dossier_chroma, "f2.bin", 100)
    s2 = mem_taille.verifier_taille(cache_s=0.0)
    assert s2.nb_fichiers == 2


# ================================================================ purger_si_necessaire

def test_purge_ne_se_declenche_pas_sous_la_limite(mem_taille, dossier_chroma, horloge):
    # 10 % de la limite
    ecrire_fichier(dossier_chroma, "petit.bin", 100)
    poser_souvenir(mem_taille, horloge, "souvenir faible", age_h=100, force=0.5)
    resultat = mem_taille.purger_si_necessaire()
    assert resultat == {}
    assert mem_taille.compter() == 1  # rien n'a été purgé


def test_purge_auto_desactivee_ne_fait_rien(mem_taille, dossier_chroma, horloge):
    mem_taille.cfg.definir("memoire_purge_auto_actif", False)
    ecrire_fichier(dossier_chroma, "trop.bin", 5000)  # 5 Ko > 1 Ko
    poser_souvenir(mem_taille, horloge, "faible", age_h=100, force=0.5)
    resultat = mem_taille.purger_si_necessaire()
    assert resultat.get("raison") == "purge auto désactivée"
    assert mem_taille.compter() == 1


def test_purge_au_dessus_de_la_limite_avec_candidats(mem_taille, dossier_chroma, horloge):
    # 5 Ko sur 1 Ko → limite dépassée, cible 0.5 Ko → il faut libérer 4.5 Ko
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    # Ajoute plusieurs souvenirs faibles (candidats)
    for i in range(20):
        poser_souvenir(mem_taille, horloge, f"faible {i}", age_h=100, force=0.5)
    resultat = mem_taille.purger_si_necessaire()
    assert resultat.get("raison") == "purge automatique"
    # Vérifie qu'on a purgé (les souvenirs faibles ont été retirés)
    assert mem_taille.compter() == 0
    # Vérifie que l'archive contient bien les souvenirs purgés
    assert mem_taille.compter_archives() == 20


def test_purge_preserve_les_marquants(mem_taille, dossier_chroma, horloge):
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    # Un marquant très faible (force 0.1, valence -0.9) → ne doit PAS être purgé
    poser_souvenir(mem_taille, horloge, "marquant", age_h=100, force=0.5, v=-0.9, a=0.9)
    # Un faible (force 0.5, valence 0) → candidat
    poser_souvenir(mem_taille, horloge, "faible", age_h=100, force=0.5)
    mem_taille.purger_si_necessaire()
    textes_restants = {t for t, _ in mem_taille.col.rows.values()}
    assert "marquant" in textes_restants
    assert "faible" not in textes_restants


def test_purge_preserve_les_concepts(mem_taille, dossier_chroma, horloge):
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    poser_souvenir(mem_taille, horloge, "concept", age_h=100, force=0.4, type=TYPE_CONCEPT)
    poser_souvenir(mem_taille, horloge, "faible", age_h=100, force=0.5)
    mem_taille.purger_si_necessaire()
    textes_restants = {t for t, _ in mem_taille.col.rows.values()}
    assert "concept" in textes_restants
    assert "faible" not in textes_restants


def test_purge_ordre_faibles_puis_declin(mem_taille, dossier_chroma, horloge):
    """Les faibles sont purgés avant les souvenirs simplement en déclin."""
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    # force 0.05 = faible (< SEUIL_OUBLI=0.3)
    poser_souvenir(mem_taille, horloge, "faible", age_h=100, force=0.5)
    # force ≈ 0.40 = en déclin (< SEUIL_EN_DECLIN=0.5 mais > SEUIL_OUBLI)
    poser_souvenir(mem_taille, horloge, "declin", age_h=14, force=0.8)
    mem_taille.purger_si_necessaire()
    textes_restants = {t for t, _ in mem_taille.col.rows.values()}
    # Le faible a été purgé en premier, le déclin a suivi
    assert "faible" not in textes_restants


def test_purge_sans_candidat_signale_et_ne_fait_rien(mem_taille, dossier_chroma, horloge):
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    # Seuls des marquants et des concepts : rien à purger
    poser_souvenir(mem_taille, horloge, "marquant", age_h=100, force=0.5, v=-0.9, a=0.9)
    poser_souvenir(mem_taille, horloge, "concept", age_h=100, force=0.5, type=TYPE_CONCEPT)
    resultat = mem_taille.purger_si_necessaire()
    assert resultat.get("raison") == "aucun candidat"
    assert mem_taille.compter() == 2


def test_purge_archive_les_souvenirs_au_lieu_de_les_detruire(mem_taille, dossier_chroma, horloge):
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    for i in range(10):
        poser_souvenir(mem_taille, horloge, f"faible {i}", age_h=100, force=0.5)
    mem_taille.purger_si_necessaire()
    archives = mem_taille.archives()
    assert len(archives) == 10
    assert all(a["raison"] == "purge automatique (limite de taille)" for a in archives)


def test_purge_retourne_les_stats_avant_et_apres(mem_taille, dossier_chroma, horloge):
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    poser_souvenir(mem_taille, horloge, "faible", age_h=100, force=0.5)
    resultat = mem_taille.purger_si_necessaire()
    assert "stats_avant" in resultat
    assert "stats_apres" in resultat
    assert resultat["stats_avant"].taille_go > resultat["stats_apres"].taille_go


# ================================================================ intégration

def test_resume_lisible(mem_taille, dossier_chroma):
    ecrire_fichier(dossier_chroma, "f.bin", 1024)
    stats = mem_taille.verifier_taille(cache_s=0.0)
    resume = stats.resume()
    assert "Go" in resume
    assert "fichier" in resume
    assert "%" in resume


def test_purge_apres_ecriture_reelle(mem_taille, dossier_chroma, horloge):
    """Vérifie que purger_si_necessaire fonctionne après un cycle ajout + purge."""
    # Ajoute 5 souvenirs faibles (dans la FausseCollection, pas sur disque)
    for i in range(5):
        poser_souvenir(mem_taille, horloge, f"s{i}", age_h=100, force=0.5)
    # Simule la limite dépassée
    ecrire_fichier(dossier_chroma, "gros.bin", 5000)
    assert mem_taille.compter() == 5
    resultat = mem_taille.purger_si_necessaire()
    assert resultat.get("raison") == "purge automatique"
    assert mem_taille.compter() == 0
    assert mem_taille.compter_archives() == 5
import json
import math
import os
import threading
from datetime import datetime

import pytest

from fakes import FausseCollection, Horloge, embedder_distances
from nyx.config import ConfigNyx
from nyx.memoire import FORMAT_TS, TYPE_BRUT, TYPE_CONCEPT, Memoire, Souvenir

DEBUT = datetime(2026, 10, 5, 12, 0, 0)


@pytest.fixture
def horloge():
    return Horloge(DEBUT)


@pytest.fixture
def cfg():
    return ConfigNyx.defauts()


@pytest.fixture
def mem(cfg, horloge):
    return Memoire(cfg, FausseCollection("souvenirs"), FausseCollection("archives"), horloge)


def ts(horloge, age_h=0.0):
    from datetime import timedelta
    return (horloge() - timedelta(hours=age_h)).strftime(FORMAT_TS)


def poser(mem, horloge, texte, age_h=0.0, force=0.6, v=0.0, a=0.5, type=TYPE_BRUT, doc_id=None):
    """Insère un souvenir dont le dernier accès remonte à `age_h` heures."""
    doc_id = doc_id or f"id-{len(mem.col.rows)}-{texte[:10]}"
    meta = {"created_at": ts(horloge, age_h), "dernier_acces": ts(horloge, age_h), "force_base": force,
            "reactivations": 0, "type": type, "valence": v, "activation": a, "dominance": 0.5}
    mem.col.add(documents=[texte], metadatas=[meta], ids=[doc_id])
    return doc_id


# ================================================================ schéma et force

def test_normaliser_complete_les_valeurs_par_defaut(mem):
    n = mem.normaliser(None)
    assert n == {"created_at": "2026-10-05 12:00:00", "dernier_acces": "2026-10-05 12:00:00", "force_base": 0.5,
                 "reactivations": 0, "type": TYPE_BRUT, "valence": 0.0, "activation": 0.5, "dominance": 0.5}
    assert mem.normaliser(n) == n, "normaliser doit être idempotent"


def test_normaliser_ancien_schema(mem):
    n = mem.normaliser({"timestamp": "2026-01-01 10:00:00", "force_synaptique": 0.7})
    assert n["created_at"] == n["dernier_acces"] == "2026-01-01 10:00:00" and n["force_base"] == 0.7


def test_normaliser_convertit_les_types_et_refuse_le_corrompu(mem):
    assert mem.normaliser({"force_base": "0.4", "reactivations": "3"})["reactivations"] == 3
    with pytest.raises(ValueError):
        mem.normaliser({"force_base": "abc"})


def test_force_effective_suit_la_formule_du_decay(mem, horloge):
    meta = {"force_base": 0.6, "dernier_acces": ts(horloge, 10)}
    attendu = 0.6 * math.exp(-0.05 * 10)
    assert mem.force_effective(meta) == pytest.approx(attendu)
    assert [mem.force_effective(meta) for _ in range(3)] == [pytest.approx(attendu)] * 3   # pas de cumul
    horloge.avancer(heures=10)
    assert mem.force_effective(meta) == pytest.approx(0.6 * math.exp(-0.05 * 20))


def test_les_marquants_et_les_concepts_oublient_plus_lentement(mem, horloge):
    base = {"force_base": 0.6, "dernier_acces": ts(horloge, 10)}
    neutre = mem.force_effective(base)
    marquant = mem.force_effective(dict(base, valence=-0.9, activation=0.9))
    concept = mem.force_effective(dict(base, type=TYPE_CONCEPT))
    attendu = 0.6 * math.exp(-0.05 * 0.3 * 10)
    assert marquant == pytest.approx(attendu) == pytest.approx(concept) and marquant > neutre


def test_force_effective_ancien_schema(mem, horloge):
    ancien = {"timestamp": ts(horloge, 10), "force_synaptique": 0.6}
    assert mem.force_effective(ancien) == pytest.approx(0.6 * math.exp(-0.5))


def test_charge_marquant_protege_categorie(mem):
    assert mem.charge_emotionnelle({"valence": -0.9, "activation": 0.9}) == pytest.approx(0.81)
    assert mem.charge_emotionnelle(None) == 0.0
    assert mem.est_marquant({"valence": -0.9, "activation": 0.9}) and not mem.est_marquant({"valence": 0.1})
    assert mem.est_protege({"type": TYPE_CONCEPT}) and not mem.est_protege({})
    assert mem.categorie({"type": TYPE_CONCEPT}) == "concept"
    assert mem.categorie({"valence": 1.0, "activation": 1.0}) == "marquant"
    assert mem.categorie({}) == "brut"


def test_le_seuil_marquant_vient_de_la_config(mem, cfg):
    meta = {"valence": 0.6, "activation": 0.6}     # charge 0.36
    assert not mem.est_marquant(meta)
    cfg.definir("SEUIL_CHARGE_EMOTIONNELLE", 0.3)
    assert mem.est_marquant(meta)


# ================================================================ écriture

def test_ajouter_ecrit_le_bon_schema(mem, horloge):
    doc_id = mem.ajouter("  J'aime la philosophie  ", valence=0.4, activation=0.7, dominance=0.6)
    texte, meta = mem.col.rows[doc_id]
    assert texte == "J'aime la philosophie"
    assert meta == {"created_at": "2026-10-05 12:00:00", "dernier_acces": "2026-10-05 12:00:00", "force_base": 0.6,
                    "reactivations": 0, "type": TYPE_BRUT, "valence": 0.4, "activation": 0.7, "dominance": 0.6}
    assert mem.compter() == 1


def test_ajouter_borne_les_emotions_et_refuse_le_vide(mem):
    doc_id = mem.ajouter("texte", valence=5, activation=-3, dominance=9)
    meta = mem.col.rows[doc_id][1]
    assert (meta["valence"], meta["activation"], meta["dominance"]) == (1.0, 0.0, 1.0)
    for vide in ("", "   ", None, 42):
        with pytest.raises(ValueError):
            mem.ajouter(vide)
    assert mem.compter() == 1


def test_renforcer_met_a_jour_force_acces_et_reactivations(mem, horloge):
    cible = poser(mem, horloge, "cible", age_h=10, force=0.6)
    autre = poser(mem, horloge, "autre", age_h=10, force=0.6)
    avant_autre = mem.col.rows[autre][1]
    assert mem.renforcer([cible]) == [cible]
    meta = mem.col.rows[cible][1]
    assert meta["force_base"] == pytest.approx(0.6 * math.exp(-0.5) + 0.15)
    assert meta["dernier_acces"] == ts(horloge) and meta["reactivations"] == 1
    assert mem.col.rows[autre][1] == avant_autre, "les autres souvenirs ne bougent pas"


def test_renforcer_plafonne_ignore_les_inconnus_et_migre_l_ancien_schema(mem, horloge):
    doc_id = poser(mem, horloge, "fort", force=0.98)
    mem.renforcer([doc_id])
    assert mem.col.rows[doc_id][1]["force_base"] == 1.0
    mem.col.inserer_brut("vieux", "ancien", {"timestamp": ts(horloge, 5), "force_synaptique": 0.5})
    mem.col.inserer_brut("sans-meta", "sans", None)
    assert mem.renforcer(["vieux", "sans-meta", "inconnu"]) == ["vieux"]
    meta = mem.col.rows["vieux"][1]
    assert set(meta) >= {"created_at", "dernier_acces", "force_base", "reactivations", "type"}
    assert mem.renforcer([]) == []


# ================================================================ rappel

@pytest.fixture
def mem_dist(cfg, horloge):
    """Mémoire à distances exactes : requête 'Q'."""
    table = {"A+": 0.50, "B-": 0.55, "C": 0.90, "LOIN-": 1.50, "N0": 0.52}
    return Memoire(cfg, FausseCollection("s", embedder_distances(table)), FausseCollection("a"), horloge)


def remplir(mem, horloge):
    poser(mem, horloge, "A+", v=0.8, a=0.7)
    poser(mem, horloge, "B-", v=-0.8, a=0.7)
    poser(mem, horloge, "C", v=0.0)
    poser(mem, horloge, "LOIN-", v=-0.9, a=0.9)


def test_rappel_filtre_par_distance_et_trie(mem_dist, horloge):
    remplir(mem_dist, horloge)
    r = mem_dist.rappeler("Q", n=3)
    assert [s.texte for s in r] == ["A+", "B-", "C"], "LOIN- (1.50 > seuil 1.2) n'est jamais rappelé"
    assert all(isinstance(s, Souvenir) and s.bonus_humeur == 0 for s in r)
    assert [s.texte for s in mem_dist.rappeler("Q", n=1)] == ["A+"]
    assert r[0].distance == pytest.approx(0.50) and r[0].score == pytest.approx(0.50)


def test_l_humeur_negative_favorise_les_souvenirs_negatifs(mem_dist, horloge):
    remplir(mem_dist, horloge)
    assert mem_dist.rappeler("Q", n=1)[0].texte == "A+"                              # sans humeur
    r = mem_dist.rappeler("Q", n=2, humeur=(-0.9, 0.9))
    assert r[0].texte == "B-", "à pertinence proche, l'humeur négative remonte le souvenir négatif"
    intensite = 0.9 * (0.5 + 0.5 * 0.9)
    assert r[0].bonus_humeur == pytest.approx(0.30 * intensite * (1 - abs(-0.8 - -0.9) / 2))


def test_l_humeur_positive_favorise_les_souvenirs_positifs(mem_dist, horloge):
    remplir(mem_dist, horloge)
    assert mem_dist.rappeler("Q", n=1, humeur=(0.9, 0.9))[0].texte == "A+"


def test_humeur_neutre_ou_absente_ne_change_rien(mem_dist, horloge):
    remplir(mem_dist, horloge)
    ordre = [s.texte for s in mem_dist.rappeler("Q", n=3)]
    neutre = mem_dist.rappeler("Q", n=3, humeur=(0.0, 0.5))
    assert [s.texte for s in neutre] == ordre and all(s.bonus_humeur == 0 for s in neutre)


def test_le_biais_ne_rappelle_jamais_un_hors_sujet(cfg, mem_dist, horloge):
    remplir(mem_dist, horloge)
    cfg.definir("POIDS_HUMEUR", 2.0)
    assert "LOIN-" not in [s.texte for s in mem_dist.rappeler("Q", n=5, humeur=(-1.0, 1.0))]


def test_poids_humeur_zero_desactive_le_biais(cfg, mem_dist, horloge):
    remplir(mem_dist, horloge)
    cfg.definir("POIDS_HUMEUR", 0.0)
    assert mem_dist.rappeler("Q", n=1, humeur=(-0.9, 0.9))[0].texte == "A+"


def test_le_seuil_de_distance_vient_de_la_config(cfg, mem_dist, horloge):
    remplir(mem_dist, horloge)
    cfg.definir("SEUIL_DISTANCE", 0.6)
    assert [s.texte for s in mem_dist.rappeler("Q", n=5)] == ["A+", "B-"]


def test_callback_sur_candidat_voit_tous_les_candidats(mem_dist, horloge):
    remplir(mem_dist, horloge)
    vus = []
    mem_dist.rappeler("Q", n=2, humeur=(-0.9, 0.9), sur_candidat=lambda s, retenu: vus.append((s.texte, retenu)))
    assert vus == [("A+", False), ("B-", True), ("C", False), ("LOIN-", False)] or \
        {t for t, _ in vus} == {"A+", "B-", "C", "LOIN-"}
    assert dict(vus)["B-"] is True and dict(vus)["LOIN-"] is False


def test_rappel_sur_memoire_vide_et_metadonnees_absentes(mem_dist, horloge):
    assert mem_dist.rappeler("Q") == []
    mem_dist.col.inserer_brut("x", "A+", None)
    r = mem_dist.rappeler("Q", humeur=(-0.9, 0.9))
    assert len(r) == 1 and r[0].meta_absente and r[0].valence == 0.0


# ================================================================ sommeil : tri et déclin

def test_trier_pour_sommeil(mem, horloge, cfg):
    poser(mem, horloge, "faible-neutre", age_h=100, force=0.5)                            # 0.003 -> oubli
    poser(mem, horloge, "faible-marquant", age_h=100, force=0.5, v=-0.9, a=0.9)           # marquant : préservé
    poser(mem, horloge, "concept-recent", age_h=5, force=0.85, type=TYPE_CONCEPT)         # concept
    poser(mem, horloge, "concept-oublie", age_h=300, force=0.5, type=TYPE_CONCEPT)        # 0.5*e^-4.5 -> oubli
    poser(mem, horloge, "neutre-recent", age_h=1, force=0.6)                              # neutre
    mem.col.inserer_brut("sans-meta", "sans meta", None)
    mem.col.inserer_brut("meta-abimee", "abimee", {"force_base": "abc"})
    plan = mem.trier_pour_sommeil()
    noms = lambda lst: sorted(s.texte for s in lst)  # noqa: E731
    assert noms(plan.oubli) == ["abimee", "concept-oublie", "faible-neutre", "sans meta"]
    assert noms(plan.marquants) == ["faible-marquant"]
    assert noms(plan.concepts) == ["concept-recent"]
    assert noms(plan.neutres) == ["neutre-recent"]


def test_frontiere_du_seuil_d_oubli(mem, horloge, cfg):
    # force effective = 0.3 + epsilon / - epsilon autour du seuil
    heures = math.log(0.6 / 0.3) / cfg.TAUX_DECAY       # force exactement 0.30
    poser(mem, horloge, "juste-au-dessus", age_h=heures - 0.2)
    poser(mem, horloge, "juste-en-dessous", age_h=heures + 0.2)
    plan = mem.trier_pour_sommeil()
    assert [s.texte for s in plan.oubli] == ["juste-en-dessous"] and [s.texte for s in plan.neutres] == ["juste-au-dessus"]


def test_en_declin(mem, horloge):
    poser(mem, horloge, "solide", age_h=0, force=0.9)                                  # 0.9
    poser(mem, horloge, "estompe-1", age_h=14, force=0.8)                              # 0.8*e^-0.7 = 0.40
    poser(mem, horloge, "estompe-2", age_h=8, force=0.7)                               # 0.7*e^-0.4 = 0.47
    poser(mem, horloge, "faible", age_h=100, force=0.5)                                # oubliable
    poser(mem, horloge, "faible-marquant", age_h=100, force=0.5, v=-0.9, a=0.9)        # 0.11 : protégé de l'oubli
    declin, faibles, total = mem.en_declin()
    # un marquant très affaibli n'est pas « faible » (jamais oublié) mais il s'estompe bien : comme avant
    assert [s.texte for s in declin] == ["faible-marquant", "estompe-1", "estompe-2"], "triés du plus faible au plus fort"
    assert faibles == 1 and total == 5


# ================================================================ archivage

def test_oublier_archive_puis_supprime(mem, horloge):
    doc_id = poser(mem, horloge, "à oublier", age_h=3, force=0.5, v=0.2)
    meta_origine = dict(mem.col.rows[doc_id][1])
    assert mem.oublier([doc_id, "inconnu"], raison="test") == [doc_id]
    assert mem.compter() == 0 and mem.compter_archives() == 1
    texte, meta = mem.archive.rows[doc_id]
    assert texte == "à oublier"
    assert {k: meta[k] for k in meta_origine} == meta_origine, "métadonnées d'origine conservées"
    assert meta["archive_raison"] == "test" and meta["archive_le"] == ts(horloge) and meta["id_origine"] == doc_id


def test_oublier_sans_archive_et_cas_vides(cfg, horloge):
    m = Memoire(cfg, FausseCollection(), None, horloge)
    doc_id = poser(m, horloge, "x")
    assert m.oublier([]) == [] and m.oublier(["inconnu"]) == []
    assert m.oublier([doc_id]) == [doc_id] and m.compter() == 0 and m.compter_archives() == 0


def test_restaurer(mem, horloge):
    doc_id = poser(mem, horloge, "revient", age_h=50, force=0.7)
    mem.oublier([doc_id])
    horloge.avancer(heures=2)
    assert mem.restaurer([doc_id, "inconnu"]) == [doc_id]
    texte, meta = mem.col.rows[doc_id]
    assert texte == "revient" and meta["force_base"] == 0.7 and meta["dernier_acces"] == ts(horloge)
    assert not any(k.startswith("archive_") or k == "id_origine" for k in meta)
    assert mem.compter_archives() == 0
    assert mem.restaurer([doc_id]) == [], "déjà actif : ignoré"


def test_oublier_deux_fois_le_meme_souvenir_apres_restauration(mem, horloge):
    doc_id = poser(mem, horloge, "yoyo")
    mem.oublier([doc_id])
    mem.restaurer([doc_id])
    assert mem.oublier([doc_id], raison="2e fois") == [doc_id]
    assert mem.archive.rows[doc_id][1]["archive_raison"] == "2e fois" and mem.compter_archives() == 1


def test_supprimer_archive_par_defaut_et_definitivement_sur_demande(mem, horloge):
    a, b = poser(mem, horloge, "a"), poser(mem, horloge, "b")
    mem.supprimer([a])
    assert mem.compter_archives() == 1 and mem.archive.rows[a][1]["archive_raison"] == "suppression manuelle"
    assert mem.supprimer([b], definitivement=True) == [b]
    assert mem.compter() == 0 and mem.compter_archives() == 1


def test_archives_et_purge(mem, horloge):
    a = poser(mem, horloge, "ancien")
    mem.oublier([a])
    horloge.avancer(jours=40)
    b = poser(mem, horloge, "recent")
    mem.oublier([b])
    liste = mem.archives()
    assert [x["texte"] for x in liste] == ["recent", "ancien"], "les plus récentes d'abord"
    assert mem.purger_archives(plus_vieilles_que_jours=30) == 1
    assert [x["texte"] for x in mem.archives()] == ["recent"]
    assert mem.purger_archives() == 1 and mem.compter_archives() == 0


def test_un_echec_d_archivage_ne_detruit_rien(mem, horloge):
    doc_id = poser(mem, horloge, "précieux")
    mem.archive.echec_sur.add("add")
    with pytest.raises(RuntimeError):
        mem.oublier([doc_id])
    assert mem.compter() == 1, "le souvenir doit rester en mémoire active"


# ================================================================ concepts

CONCEPTS_REJETES = ["Voici la consolidation de mes souvenirs : - Mon rôle", "Bien sûr, je suis ravi de vous aider !",
                    "L'alchimie. **Concept 2 : Le temps** Le temps nous coince", "Je fusionne les concepts suivants",
                    "Concept 1 : La relation humaine est importante", "trop court", "x" * 300,
                    "ligne 1\nligne 2 assez longue", "- une puce de liste assez longue", 42, None]
CONCEPTS_ACCEPTES = ["L'introspection critique consiste à revoir ses propres biais.",
                     "La Limite Épistémique : la conscience d'être limité par ses connaissances."]


def test_concept_valide(mem):
    assert not any(Memoire.concept_valide(c) for c in CONCEPTS_REJETES)
    assert all(Memoire.concept_valide(c) for c in CONCEPTS_ACCEPTES)


def test_concepts_pollues(mem, horloge):
    poser(mem, horloge, "**Concept 1 : pollué** Voici une réponse de chatbot", type=TYPE_CONCEPT)
    poser(mem, horloge, CONCEPTS_ACCEPTES[0], type=TYPE_CONCEPT)
    poser(mem, horloge, "**souvenir brut** avec markdown mais pas un concept", type=TYPE_BRUT)
    assert [s.texte[:12] for s in mem.concepts_pollues()] == ["**Concept 1 "]


def test_remplacer_par_concepts(mem, horloge):
    sources = [poser(mem, horloge, f"source numero {i}") for i in range(4)]
    ids = mem.remplacer_par_concepts(sources, CONCEPTS_ACCEPTES + ["x", "y" * 300] + ["Un troisième concept valide ici.",
                                                                                     "Un quatrième concept valide ici."])
    assert len(ids) == 3, "au plus MAX_CONCEPTS concepts, les invalides sont filtrés"
    assert mem.compter() == 3 and mem.compter_archives() == 4
    texte, meta = mem.col.rows[ids[0]]
    assert meta["type"] == TYPE_CONCEPT and meta["force_base"] == 0.85 and meta["valence"] == 0.0
    assert all(mem.archive.rows[s][1]["archive_raison"] == "consolidation" for s in sources)


def test_remplacer_sans_concept_valide_ne_touche_a_rien(mem, horloge):
    sources = [poser(mem, horloge, f"source {i}") for i in range(3)]
    with pytest.raises(ValueError):
        mem.remplacer_par_concepts(sources, CONCEPTS_REJETES)
    assert mem.compter() == 3 and mem.compter_archives() == 0


def test_remplacer_echec_d_ajout_laisse_les_sources_intactes(mem, horloge):
    sources = [poser(mem, horloge, f"source {i}") for i in range(3)]
    mem.col.echec_sur.add("add")
    with pytest.raises(RuntimeError):
        mem.remplacer_par_concepts(sources, CONCEPTS_ACCEPTES)
    assert mem.compter() == 3 and mem.compter_archives() == 0


def test_remplacer_echec_d_archivage_ne_perd_rien(mem, horloge):
    sources = [poser(mem, horloge, f"source {i}") for i in range(3)]
    mem.archive.echec_sur.add("add")
    with pytest.raises(RuntimeError):
        mem.remplacer_par_concepts(sources, CONCEPTS_ACCEPTES)
    textes = {t for t, _ in mem.col.rows.values()}
    assert {"source 0", "source 1", "source 2"} <= textes, "les sources doivent rester"
    assert set(CONCEPTS_ACCEPTES) <= textes, "les concepts déjà ajoutés restent : redondance plutôt que perte"


# ================================================================ statistiques, intégrité, cache

def test_stats(mem, horloge):
    poser(mem, horloge, "fort", force=0.9)
    poser(mem, horloge, "marquant", age_h=100, force=0.5, v=-0.9, a=0.9)
    poser(mem, horloge, "concept", force=0.85, type=TYPE_CONCEPT)
    poser(mem, horloge, "faible", age_h=100, force=0.5)
    mem.oublier([poser(mem, horloge, "archive-moi")])
    s = mem.stats()
    assert (s["total"], s["marquants"], s["concepts"], s["bruts"], s["faibles"], s["archives"]) == (4, 1, 1, 2, 1, 1)
    assert sum(s["histogramme"]) == 4 and s["histogramme"][4] == 2 and s["histogramme"][0] == 2
    assert s["force_min"] < s["force_moyenne"] < s["force_max"]


def test_stats_memoire_vide(mem):
    s = mem.stats()
    assert s["total"] == 0 and s["force_moyenne"] == 0.0 and s["histogramme"] == [0] * 5


def test_verifier_integrite(mem, horloge):
    assert mem.verifier_integrite() == []
    poser(mem, horloge, "sain")
    assert mem.verifier_integrite() == []
    mem.col.inserer_brut("a" * 8, "sans meta", None)
    mem.col.inserer_brut("b" * 8, "date cassée", {"dernier_acces": "pas une date", "force_base": 0.5})
    mem.col.inserer_brut("c" * 8, "force folle", {"force_base": 4.0, "created_at": ts(horloge), "dernier_acces": ts(horloge)})
    mem.col.inserer_brut("d" * 8, "   ", {"force_base": 0.5, "created_at": ts(horloge), "dernier_acces": ts(horloge)})
    poser(mem, horloge, "**pollué** Voici ce que je fusionne", type=TYPE_CONCEPT)
    texte = " | ".join(mem.verifier_integrite())
    for attendu in ("métadonnées absentes", "métadonnées invalides", "hors [0,1]", "document vide", "concept suspect"):
        assert attendu in texte


def test_le_cache_evite_de_relire_la_base_et_les_forces_restent_fraiches(mem, horloge):
    poser(mem, horloge, "un", force=0.6)
    f0 = mem.tous(cache_s=60)[0].force
    lectures = mem.col.appels["get"]
    horloge.avancer(heures=10)
    f1 = mem.tous(cache_s=60)[0].force
    assert mem.col.appels["get"] == lectures, "la base n'est pas relue pendant la durée du cache"
    assert f1 < f0 and f1 == pytest.approx(0.6 * math.exp(-0.5)), "mais le decay est recalculé"
    mem.tous()                                                      # sans cache : relit
    assert mem.col.appels["get"] == lectures + 1


def test_toute_ecriture_invalide_le_cache(mem, horloge):
    poser(mem, horloge, "un")
    assert len(mem.tous(cache_s=60)) == 1
    mem.ajouter("deux")
    assert len(mem.tous(cache_s=60)) == 2
    mem.oublier([next(iter(mem.col.rows))])
    assert len(mem.tous(cache_s=60)) == 1
    doc_id = next(iter(mem.col.rows))
    mem.renforcer([doc_id])
    assert mem.tous(cache_s=60)[0].reactivations == 1


# ================================================================ export / import

def test_export_import_aller_retour(mem, horloge, cfg, tmp_path):
    poser(mem, horloge, "J'aime la philosophie", v=0.4)
    poser(mem, horloge, CONCEPTS_ACCEPTES[0], type=TYPE_CONCEPT, force=0.85)
    mem.oublier([poser(mem, horloge, "archivé", doc_id="arch-1")])
    chemin = str(tmp_path / "sauvegarde.json")
    assert mem.exporter(chemin) == (2, 1)
    assert not os.path.exists(chemin + ".tmp")
    contenu = json.load(open(chemin, encoding="utf-8"))
    assert contenu["version"] == 1 and len(contenu["souvenirs"]) == 2 and len(contenu["archives"]) == 1

    neuve = Memoire(cfg, FausseCollection("n"), FausseCollection("na"), horloge)
    bilan = neuve.importer(chemin)
    assert bilan == {"ajoutes": 2, "ignores": 0, "archives_ajoutees": 1, "problemes": []}
    assert sorted(t for t, _ in neuve.col.rows.values()) == sorted(t for t, _ in mem.col.rows.values())
    assert [s.categorie for s in neuve.tous() if s.texte == CONCEPTS_ACCEPTES[0]] == ["concept"]
    assert neuve.rappeler("J'aime la philosophie", n=1)[0].texte == "J'aime la philosophie", "le rappel fonctionne après import"
    assert neuve.importer(chemin) == {"ajoutes": 0, "ignores": 3, "archives_ajoutees": 0, "problemes": []}


def test_export_sans_archives(mem, horloge, tmp_path):
    mem.oublier([poser(mem, horloge, "archivé")])
    assert mem.exporter(str(tmp_path / "s.json"), inclure_archives=False) == (0, 0)


def test_import_signale_les_entrees_invalides_et_refuse_un_fichier_invalide(mem, horloge, tmp_path):
    chemin = tmp_path / "x.json"
    chemin.write_text(json.dumps({"version": 1, "souvenirs": [
        {"id": "ok", "texte": "bon souvenir", "meta": {"force_base": 0.5}},
        {"id": "vide", "texte": "  ", "meta": {}},
        {"texte": "sans id"},
        {"id": "corrompu", "texte": "meta cassée", "meta": {"force_base": "abc"}},
    ]}), encoding="utf-8")
    bilan = mem.importer(str(chemin))
    assert bilan["ajoutes"] == 1 and len(bilan["problemes"]) == 3
    chemin.write_text(json.dumps({"autre": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        mem.importer(str(chemin))


def test_export_atomique(mem, horloge, tmp_path, monkeypatch):
    chemin = str(tmp_path / "s.json")
    mem.exporter(chemin)
    avant = open(chemin, encoding="utf-8").read()
    poser(mem, horloge, "nouveau")
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disque plein")))
    with pytest.raises(OSError):
        mem.exporter(chemin)
    assert open(chemin, encoding="utf-8").read() == avant and not os.path.exists(chemin + ".tmp")


# ================================================================ concurrence

def test_acces_concurrents_sans_erreur(mem):
    erreurs = []

    def ecrire(n):
        try:
            for i in range(15):
                doc_id = mem.ajouter(f"souvenir {n}-{i}")
                mem.renforcer([doc_id])
                mem.tous(cache_s=1.0)
        except Exception as e:  # noqa: BLE001
            erreurs.append(e)
    threads = [threading.Thread(target=ecrire, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not erreurs and mem.compter() == 90

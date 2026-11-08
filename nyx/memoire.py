"""
nyx/memoire.py : l'hippocampe de Nyx (module 3/9).

Remplaçait l'ancien code : _normaliser_meta, _force_effective, _renforcer_souvenirs,
_rappeler_souvenirs, _souvenirs_en_declin, _concept_valide, le tri de _consolider_memoire.

Fonctionnalités :
  - ARCHIVAGE : un souvenir oublié/supprimé/consolidé est déplacé dans une archive
    restaurable.
  - CACHE de lecture pour ne pas relire la base à chaque tick.
  - EXPORT / IMPORT JSON.
  - INTÉGRITÉ : détection des métadonnées abîmées ou des concepts pollués.
  - EMBEDDINGS EXTERNES : nomic-embed-text via OllamaEmbeddingFunction.
  - LIMITE DE TAILLE : surveillance + purge automatique.
  - NOUVEAU (v6) : support du type `souvenir_onirique` (rêves marquants).

Types de souvenirs :
  - "souvenir_brut"       : souvenir ordinaire
  - "concept_consolidé"   : fusion de souvenirs neutres
  - "souvenir_onirique"   : rêve marquant (protégé comme un marquant)

Règles de protection :
  - Les rêves oniriques sont traités comme des marquants (jamais oubliés)
    mais avec un decay plus rapide que les concepts (ils s'estompent comme
    de vrais rêves).
"""
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

_log = logging.getLogger("nyx.memoire")

FORMAT_TS = "%Y-%m-%d %H:%M:%S"
TYPE_BRUT = "souvenir_brut"
TYPE_CONCEPT = "concept_consolidé"
TYPE_ONIRIQUE = "souvenir_onirique"
VERSION_EXPORT = 2
_PREFIXES_REFUSES = ("voici", "bien sûr", "je fusionne", "je suis ravi", "concept ")


@dataclass(frozen=True)
class Souvenir:
    """Un souvenir lu en mémoire."""
    id: str
    texte: str
    created_at: str
    dernier_acces: str
    force_base: float
    force: float
    reactivations: int
    type: str
    valence: float
    activation: float
    dominance: float
    categorie: str            # 'marquant' | 'concept' | 'onirique' | 'brut'
    faible: bool
    meta_absente: bool = False
    meta_invalide: bool = False
    distance: float = None
    bonus_humeur: float = 0.0

    @property
    def score(self):
        return None if self.distance is None else self.distance - self.bonus_humeur


@dataclass
class PlanSommeil:
    """Classement de la mémoire avant le sommeil."""
    oubli: list
    marquants: list
    concepts: list
    oniriques: list
    neutres: list


@dataclass
class StatsTaille:
    taille_go: float
    limite_go: float
    fraction: float
    nb_fichiers: int
    au_dessus_avertissement: bool
    au_dessus_limite: bool

    def resume(self):
        return (f"{self.taille_go:.3f} Go / {self.limite_go:.2f} Go "
                f"({self.fraction * 100:.1f} %) · {self.nb_fichiers} fichier(s)")


class Memoire:
    FORCE_INITIALE = 0.6
    FORCE_CONCEPT = 0.85
    FORCE_ONIRIQUE = 0.7
    MAX_CONCEPTS = 3

    def __init__(self, cfg, collection, archive=None, horloge=None,
                 embedding_function=None, chemin_db=None):
        self.cfg = cfg
        self.col = collection
        self.archive = archive
        self._maintenant = horloge or datetime.now
        self.ef = embedding_function
        self.chemin_db = chemin_db or getattr(cfg, "chemin_db", None)
        self._verrou = threading.RLock()
        self._cache = None
        self._derniere_verif_taille = 0.0

    # ------------------------------------------------------------------ temps

    def horodatage(self):
        return self._maintenant().strftime(FORMAT_TS)

    def heures_depuis(self, ts):
        try:
            dt = datetime.strptime(ts, FORMAT_TS)
            return max(0.0, (self._maintenant() - dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------ schéma

    def normaliser(self, meta):
        meta = meta or {}
        ancien_ts = meta.get("timestamp", self.horodatage())
        return {
            "created_at": meta.get("created_at", ancien_ts),
            "dernier_acces": meta.get("dernier_acces", ancien_ts),
            "force_base": float(meta.get("force_base", meta.get("force_synaptique", 0.5))),
            "reactivations": int(meta.get("reactivations", 0)),
            "type": meta.get("type", TYPE_BRUT),
            "valence": float(meta.get("valence", 0.0)),
            "activation": float(meta.get("activation", 0.5)),
            "dominance": float(meta.get("dominance", 0.5)),
        }

    @staticmethod
    def charge_emotionnelle(meta):
        if meta is None:
            return 0.0
        return abs(float(meta.get("valence", 0.0))) * float(meta.get("activation", 0.5))

    def est_marquant(self, meta):
        return self.charge_emotionnelle(meta) >= self.cfg.SEUIL_CHARGE_EMOTIONNELLE

    def est_onirique(self, meta):
        if meta is None:
            return False
        return meta.get("type") == TYPE_ONIRIQUE

    def est_protege(self, meta):
        """
        Protection à l'oubli :
          - marquants (charge émotionnelle ≥ seuil)
          - concepts consolidés
          - souvenirs oniriques (rêves marquants)
        """
        m = self.normaliser(meta)
        return (self.est_marquant(m)
                or m["type"] == TYPE_CONCEPT
                or m["type"] == TYPE_ONIRIQUE)

    def categorie(self, meta):
        m = self.normaliser(meta)
        if m["type"] == TYPE_CONCEPT:
            return "concept"
        if m["type"] == TYPE_ONIRIQUE:
            return "onirique"
        return "marquant" if self.est_marquant(m) else "brut"

    def force_effective(self, meta):
        """
        Decay PARESSEUX : calculé à la lecture depuis force_base et le temps écoulé.

        Facteurs de protection :
          - marquants          : 0.3× la vitesse de decay normale
          - concepts           : 0.3× la vitesse de decay normale
          - oniriques          : 0.5× la vitesse (plus rapide que concepts, plus lent que brut)
        """
        m = self.normaliser(meta)
        if m["type"] == TYPE_ONIRIQUE:
            facteur = 0.5
        elif self.est_protege(m):
            facteur = self.cfg.FACTEUR_DECAY_PROTEGE
        else:
            facteur = 1.0
        taux = self.cfg.TAUX_DECAY * facteur
        return m["force_base"] * math.exp(-taux * self.heures_depuis(m["dernier_acces"]))

    def _construire(self, doc_id, doc, meta, distance=None, bonus=0.0):
        absente, invalide = meta is None, False
        try:
            n = self.normaliser(meta)
            force = self.force_effective(n)
            categorie = self.categorie(n)
            marquant = self.est_marquant(n)
            onirique = self.est_onirique(n)
        except (ValueError, TypeError):
            invalide = True
            n, force, categorie, marquant, onirique = self.normaliser(None), 0.0, "brut", False, False
        return Souvenir(
            id=doc_id, texte=doc or "", created_at=n["created_at"], dernier_acces=n["dernier_acces"],
            force_base=n["force_base"], force=force, reactivations=n["reactivations"], type=n["type"],
            valence=n["valence"], activation=n["activation"], dominance=n["dominance"],
            categorie=categorie,
            faible=(force < self.cfg.SEUIL_OUBLI and not marquant and not onirique),
            meta_absente=absente, meta_invalide=invalide, distance=distance, bonus_humeur=bonus)

    # ------------------------------------------------------------------ embeddings

    def _embed_documents(self, textes):
        if self.ef is None:
            return None
        return self.ef.embed_documents(textes)

    def _embed_query(self, texte):
        if self.ef is None:
            return None
        return self.ef.embed_query(texte)

    # ------------------------------------------------------------------ lecture

    def compter(self):
        return self.col.count()

    def _lire_brut(self, cache_s=0.0):
        if cache_s and self._cache and time.monotonic() - self._cache[0] < cache_s:
            return self._cache[1]
        if self.col.count() == 0:
            lignes = []
        else:
            d = self.col.get()
            lignes = list(zip(d["ids"], d["documents"], d["metadatas"]))
        self._cache = (time.monotonic(), lignes)
        return lignes

    def tous(self, cache_s=0.0):
        return [self._construire(i, d, m) for i, d, m in self._lire_brut(cache_s)]

    def _invalider(self):
        self._cache = None

    # ------------------------------------------------------------------ écriture

    def ajouter(self, texte, valence=0.0, activation=0.5, dominance=0.5,
                type=TYPE_BRUT, force=None):
        """Mémorise un souvenir. Supporte maintenant le type 'souvenir_onirique'."""
        if not isinstance(texte, str) or not texte.strip():
            raise ValueError("texte du souvenir vide")
        texte = texte.strip()
        ts = self.horodatage()

        # Force par défaut selon le type
        if force is None:
            if type == TYPE_CONCEPT:
                force = self.FORCE_CONCEPT
            elif type == TYPE_ONIRIQUE:
                force = self.FORCE_ONIRIQUE
            else:
                force = self.FORCE_INITIALE

        meta = {"created_at": ts, "dernier_acces": ts,
                "force_base": float(force),
                "reactivations": 0, "type": type,
                "valence": max(-1.0, min(1.0, float(valence))),
                "activation": max(0.0, min(1.0, float(activation))),
                "dominance": max(0.0, min(1.0, float(dominance)))}
        doc_id = str(uuid.uuid4())
        embedding = self._embed_documents([texte])
        with self._verrou:
            if embedding is not None:
                self.col.add(embeddings=embedding, documents=[texte], metadatas=[meta], ids=[doc_id])
            else:
                self.col.add(documents=[texte], metadatas=[meta], ids=[doc_id])
            self._invalider()
        return doc_id

    def renforcer(self, ids):
        if not ids:
            return []
        with self._verrou:
            donnees = self.col.get(ids=list(ids))
            maintenant = self.horodatage()
            ok, metas = [], []
            for doc_id, meta in zip(donnees["ids"], donnees["metadatas"]):
                if meta is None:
                    continue
                m = self.normaliser(meta)
                m["force_base"] = min(1.0, self.force_effective(meta) + self.cfg.BONUS_REACTIVATION)
                m["dernier_acces"] = maintenant
                m["reactivations"] += 1
                ok.append(doc_id)
                metas.append(m)
            if ok:
                self.col.update(ids=ok, metadatas=metas)
                self._invalider()
        return ok

    # ------------------------------------------------------------------ rappel

    def rappeler(self, requete, n=3, humeur=None, sur_candidat=None):
        total = self.col.count()
        if total == 0:
            return []
        n_demande = min(max(n, self.cfg.CANDIDATS_RAPPEL), total)
        embedding = self._embed_query(requete)
        if embedding is not None:
            r = self.col.query(query_embeddings=[embedding], n_results=n_demande)
        else:
            r = self.col.query(query_texts=[requete], n_results=n_demande)
        if not r["documents"] or not r["documents"][0]:
            return []
        if humeur is not None:
            valence, activation = humeur
            intensite = abs(valence) * (0.5 + 0.5 * activation)
        else:
            valence, intensite = 0.0, 0.0
        candidats = []
        for i, dist in enumerate(r["distances"][0]):
            meta = r["metadatas"][0][i]
            try:
                v_souvenir = float((meta or {}).get("valence", 0.0))
            except (TypeError, ValueError):
                v_souvenir = 0.0
            congruence = 1.0 - abs(v_souvenir - valence) / 2.0
            bonus = self.cfg.POIDS_HUMEUR * intensite * congruence
            candidats.append((dist - bonus, dist, bonus, i))
        retenus = sorted(c for c in candidats if c[1] <= self.cfg.SEUIL_DISTANCE)[:n]
        resultat = [self._construire(r["ids"][0][i], r["documents"][0][i], r["metadatas"][0][i], dist, bonus)
                    for _, dist, bonus, i in retenus]
        if sur_candidat is not None:
            gardes = {c[3] for c in retenus}
            for _, dist, bonus, i in sorted(candidats, key=lambda c: c[1]):
                sur_candidat(self._construire(r["ids"][0][i], r["documents"][0][i], r["metadatas"][0][i],
                                              dist, bonus), i in gardes)
        return resultat

    # ------------------------------------------------------------------ sommeil

    def trier_pour_sommeil(self):
        """
        Classe la mémoire pour le sommeil.
        Nouveauté : les rêves oniriques sont protégés comme les concepts.
        """
        plan = PlanSommeil([], [], [], [], [])
        for s in self.tous():
            if s.meta_absente or s.meta_invalide:
                plan.oubli.append(s)
            elif s.type == TYPE_CONCEPT:
                plan.concepts.append(s)
            elif s.type == TYPE_ONIRIQUE:
                plan.oniriques.append(s)
            elif self.est_marquant({"valence": s.valence, "activation": s.activation}):
                plan.marquants.append(s)
            elif s.faible:
                plan.oubli.append(s)
            else:
                plan.neutres.append(s)
        return plan

    def en_declin(self, cache_s=0.0):
        souvenirs = self.tous(cache_s)
        declin, faibles = [], 0
        for s in souvenirs:
            if s.meta_absente or s.meta_invalide:
                continue
            if s.faible:
                faibles += 1
            elif s.force < self.cfg.SEUIL_EN_DECLIN:
                declin.append(s)
        declin.sort(key=lambda s: s.force)
        return declin, faibles, len(souvenirs)

    # ------------------------------------------------------------------ archives

    def _archiver_et_supprimer(self, ids, raison):
        donnees = self.col.get(ids=list(ids))
        if not donnees["ids"]:
            return []
        if self.archive is not None:
            ts = self.horodatage()
            metas = []
            for doc_id, meta in zip(donnees["ids"], donnees["metadatas"]):
                m = dict(meta or {})
                m.update({"archive_le": ts, "archive_raison": raison, "id_origine": doc_id})
                metas.append(m)
            deja = self.archive.get(ids=donnees["ids"])["ids"]
            if deja:
                self.archive.delete(ids=deja)
            self.archive.add(documents=[d or "" for d in donnees["documents"]],
                             metadatas=metas, ids=donnees["ids"])
        self.col.delete(ids=donnees["ids"])
        self._invalider()
        return donnees["ids"]

    def oublier(self, ids, raison="oubli actif"):
        if not ids:
            return []
        with self._verrou:
            return self._archiver_et_supprimer(ids, raison)

    def supprimer(self, ids, definitivement=False, raison="suppression manuelle"):
        if not ids:
            return []
        with self._verrou:
            if definitivement or self.archive is None:
                existants = self.col.get(ids=list(ids))["ids"]
                if existants:
                    self.col.delete(ids=existants)
                    self._invalider()
                return existants
            return self._archiver_et_supprimer(ids, raison)

    def archives(self):
        if self.archive is None or self.archive.count() == 0:
            return []
        d = self.archive.get()
        lignes = [{"id": i, "texte": doc, "raison": (m or {}).get("archive_raison", ""),
                   "archive_le": (m or {}).get("archive_le", ""), "meta": m or {}}
                  for i, doc, m in zip(d["ids"], d["documents"], d["metadatas"])]
        return sorted(lignes, key=lambda x: x["archive_le"], reverse=True)

    def compter_archives(self):
        return 0 if self.archive is None else self.archive.count()

    def restaurer(self, ids):
        if self.archive is None or not ids:
            return []
        with self._verrou:
            d = self.archive.get(ids=list(ids))
            actifs = set(self.col.get(ids=d["ids"])["ids"]) if d["ids"] else set()
            restaures, docs, metas = [], [], []
            for doc_id, doc, meta in zip(d["ids"], d["documents"], d["metadatas"]):
                if doc_id in actifs:
                    continue
                propre = {k: v for k, v in (meta or {}).items()
                          if not k.startswith("archive_") and k != "id_origine"}
                try:
                    m = self.normaliser(propre)
                except (ValueError, TypeError):
                    m = self.normaliser(None)
                m["dernier_acces"] = self.horodatage()
                restaures.append(doc_id)
                docs.append(doc)
                metas.append(m)
            if restaures:
                embeddings = self._embed_documents(docs) if self.ef is not None else None
                if embeddings is not None:
                    self.col.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=restaures)
                else:
                    self.col.add(documents=docs, metadatas=metas, ids=restaures)
                self.archive.delete(ids=restaures)
                self._invalider()
            return restaures

    def purger_archives(self, plus_vieilles_que_jours=None):
        if self.archive is None or self.archive.count() == 0:
            return 0
        d = self.archive.get()
        limite = None if plus_vieilles_que_jours is None else self._maintenant() - timedelta(days=plus_vieilles_que_jours)
        a_purger = []
        for doc_id, meta in zip(d["ids"], d["metadatas"]):
            if limite is None:
                a_purger.append(doc_id)
                continue
            try:
                if datetime.strptime((meta or {}).get("archive_le", ""), FORMAT_TS) < limite:
                    a_purger.append(doc_id)
            except ValueError:
                a_purger.append(doc_id)
        if a_purger:
            self.archive.delete(ids=a_purger)
        return len(a_purger)

    # ------------------------------------------------------------------ concepts

    @staticmethod
    def concept_valide(c):
        if not isinstance(c, str):
            return False
        c = c.strip()
        if not 15 <= len(c) <= 250:
            return False
        if "**" in c or "\n" in c or c.startswith(("-", "*", "•", "#")):
            return False
        return not c.lower().startswith(_PREFIXES_REFUSES)

    def concepts_pollues(self):
        return [s for s in self.tous() if s.type == TYPE_CONCEPT and not self.concept_valide(s.texte)]

    def remplacer_par_concepts(self, ids_sources, concepts, raison="consolidation"):
        valides = [c.strip() for c in concepts if self.concept_valide(c)][:self.MAX_CONCEPTS]
        if not valides:
            raise ValueError("aucun concept valide : souvenirs conservés")
        ts = self.horodatage()
        metas = [{"created_at": ts, "dernier_acces": ts, "force_base": self.FORCE_CONCEPT, "reactivations": 0,
                  "type": TYPE_CONCEPT, "valence": 0.0, "activation": 0.5, "dominance": 0.5}
                 for _ in valides]
        ids = [str(uuid.uuid4()) for _ in valides]
        embeddings = self._embed_documents(valides)
        with self._verrou:
            if embeddings is not None:
                self.col.add(embeddings=embeddings, documents=valides, metadatas=metas, ids=ids)
            else:
                self.col.add(documents=valides, metadatas=metas, ids=ids)
            self._invalider()
            self._archiver_et_supprimer(ids_sources, raison)
        return ids

    # ------------------------------------------------------------------ stats et intégrité

    def stats(self, cache_s=0.0):
        souvenirs = self.tous(cache_s)
        forces = [s.force for s in souvenirs]
        histogramme = [0] * 5
        for f in forces:
            histogramme[max(0, min(4, int(f * 5)))] += 1
        return {
            "total": len(souvenirs),
            "marquants": sum(1 for s in souvenirs if s.categorie == "marquant"),
            "concepts": sum(1 for s in souvenirs if s.categorie == "concept"),
            "oniriques": sum(1 for s in souvenirs if s.categorie == "onirique"),
            "bruts": sum(1 for s in souvenirs if s.categorie == "brut"),
            "faibles": sum(1 for s in souvenirs if s.faible),
            "force_moyenne": sum(forces) / len(forces) if forces else 0.0,
            "force_min": min(forces) if forces else 0.0,
            "force_max": max(forces) if forces else 0.0,
            "histogramme": histogramme,
            "archives": self.compter_archives(),
        }

    def verifier_integrite(self):
        problemes = []
        for doc_id, doc, meta in self._lire_brut():
            court = doc_id[:8]
            if meta is None:
                problemes.append(f"{court} : métadonnées absentes")
                continue
            try:
                m = self.normaliser(meta)
                datetime.strptime(m["dernier_acces"], FORMAT_TS)
                datetime.strptime(m["created_at"], FORMAT_TS)
            except (ValueError, TypeError) as e:
                problemes.append(f"{court} : métadonnées invalides ({e})")
                continue
            if not 0.0 <= m["force_base"] <= 1.0:
                problemes.append(f"{court} : force_base hors [0,1] ({m['force_base']})")
            if not doc or not doc.strip():
                problemes.append(f"{court} : document vide")
            if m["type"] == TYPE_CONCEPT and not self.concept_valide(doc):
                problemes.append(f"{court} : concept suspect « {doc[:60]}... »")
        return problemes

    # ------------------------------------------------------------------ taille

    def _taille_dossier(self):
        if not self.chemin_db or not os.path.isdir(self.chemin_db):
            return 0, 0
        total = 0
        nb = 0
        for racine, _, fichiers in os.walk(self.chemin_db):
            for nom in fichiers:
                try:
                    total += os.path.getsize(os.path.join(racine, nom))
                    nb += 1
                except OSError:
                    pass
        return total, nb

    def verifier_taille(self, cache_s=30.0):
        maintenant = time.monotonic()
        if (self._derniere_verif_taille
                and maintenant - self._derniere_verif_taille[0] < cache_s):
            return self._derniere_verif_taille[1]

        octets, nb = self._taille_dossier()
        taille_go = octets / (1024 ** 3)
        limite = self.cfg.memoire_limite_go
        stats = StatsTaille(
            taille_go=taille_go,
            limite_go=limite,
            fraction=taille_go / limite if limite > 0 else 0.0,
            nb_fichiers=nb,
            au_dessus_avertissement=taille_go >= limite * self.cfg.memoire_seuil_avertissement,
            au_dessus_limite=taille_go >= limite,
        )
        self._derniere_verif_taille = (maintenant, stats)
        return stats

    def purger_si_necessaire(self):
        stats = self.verifier_taille()
        if not stats.au_dessus_limite:
            return {}
        if not self.cfg.memoire_purge_auto_actif:
            _log.warning("Mémoire au-dessus de la limite mais purge auto désactivée.")
            return {"raison": "purge auto désactivée", "stats_avant": stats}

        cible_go = stats.limite_go * self.cfg.memoire_purge_cible
        _log.warning("Purge automatique : cible %.3f Go.", cible_go)

        souvenirs = self.tous()
        # Ordre de sacrifice : faibles d'abord, puis déclin
        faibles = sorted(
            (s for s in souvenirs if s.faible and not s.meta_absente and not s.meta_invalide),
            key=lambda s: s.force,
        )
        declin = sorted(
            (s for s in souvenirs
             if not s.faible and s.force < self.cfg.SEUIL_EN_DECLIN
             and not s.meta_absente and not s.meta_invalide),
            key=lambda s: s.force,
        )
        candidats = faibles + declin

        if not candidats:
            _log.warning("Purge impossible : aucun candidat.")
            return {"raison": "aucun candidat", "stats_avant": stats}

        ids_purges = []
        for s in candidats:
            ids_purges.append(s.id)
            if len(ids_purges) >= 50:
                self.oublier(ids_purges, raison="purge automatique (limite de taille)")
                self._derniere_verif_taille = 0.0
                ids_purges = []
                nouvelles_stats = self.verifier_taille(cache_s=0.0)
                if nouvelles_stats.taille_go <= cible_go:
                    break

        if ids_purges:
            self.oublier(ids_purges, raison="purge automatique (limite de taille)")

        self._derniere_verif_taille = 0.0
        stats_apres = self.verifier_taille(cache_s=0.0)
        return {
            "raison": "purge automatique",
            "stats_avant": stats,
            "stats_apres": stats_apres,
        }

    # ------------------------------------------------------------------ export/import

    def exporter(self, chemin, inclure_archives=True):
        actifs = [{"id": i, "texte": d, "meta": m} for i, d, m in self._lire_brut()]
        archives = []
        if inclure_archives and self.archive is not None and self.archive.count():
            a = self.archive.get()
            archives = [{"id": i, "texte": d, "meta": m}
                        for i, d, m in zip(a["ids"], a["documents"], a["metadatas"])]
        contenu = {"version": VERSION_EXPORT, "exporte_le": self.horodatage(),
                   "souvenirs": actifs, "archives": archives}
        os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
        temporaire = chemin + ".tmp"
        try:
            with open(temporaire, "w", encoding="utf-8") as f:
                json.dump(contenu, f, ensure_ascii=False, indent=1)
            os.replace(temporaire, chemin)
        finally:
            if os.path.exists(temporaire):
                os.remove(temporaire)
        return len(actifs), len(archives)

    def importer(self, chemin):
        with open(chemin, encoding="utf-8") as f:
            contenu = json.load(f)
        if not isinstance(contenu, dict) or not isinstance(contenu.get("souvenirs"), list):
            raise ValueError("fichier d'export invalide")
        bilan = {"ajoutes": 0, "ignores": 0, "archives_ajoutees": 0, "problemes": []}
        with self._verrou:
            for cle, collection, compteur in (("souvenirs", self.col, "ajoutes"),
                                              ("archives", self.archive, "archives_ajoutees")):
                if collection is None:
                    continue
                for entree in contenu.get(cle, []):
                    try:
                        doc_id, texte, meta = str(entree["id"]), entree["texte"], entree.get("meta")
                        if not isinstance(texte, str) or not texte.strip():
                            raise ValueError("texte vide")
                        if collection.get(ids=[doc_id])["ids"]:
                            bilan["ignores"] += 1
                            continue
                        if cle == "souvenirs":
                            meta = self.normaliser(meta)
                        collection.add(documents=[texte], metadatas=[meta or {"importe": True}], ids=[doc_id])
                        bilan[compteur] += 1
                    except (KeyError, TypeError, ValueError) as e:
                        bilan["problemes"].append(f"entrée ignorée ({e})")
            self._invalider()
        return bilan

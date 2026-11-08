"""
Faux Chroma / Ollama pour les tests : rapides, sans modèle, sans disque.

FausseCollection est volontairement PLUS STRICTE que le vrai Chroma (vérifié sur Chroma 1.5.9), pour que
les tests attrapent plus d'erreurs que la production :
  - identique au vrai : métadonnées vides {} refusées, absence de métadonnées acceptée (renvoie None),
    liste vide refusée, delete d'un identifiant inconnu silencieux, n_results > count tronqué ;
  - plus strict que le vrai : un identifiant dupliqué LÈVE une erreur (le vrai l'ignore en silence), un
    update d'identifiant inconnu LÈVE une erreur (le vrai l'ignore), les métadonnées non scalaires sont refusées.
Ces écarts sont verrouillés par tests/test_memoire_chroma_reel.py.
"""
import collections
import copy
import math
import re
import sys
import types
import zlib
from datetime import timedelta


def embedder_par_defaut(texte, dim=256):
    """Sac de mots hashé et normalisé : textes proches = mots communs."""
    v = [0.0] * dim
    for mot in re.findall(r"\w+", texte.lower()):
        v[zlib.crc32(mot.encode()) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embedder_distances(table, requete="Q"):
    """
    Embedder 2D à distances EXACTES. La requête est placée en (1, 0) ; chaque texte de `table` est placé de
    sorte que sa distance L2² à la requête soit la valeur demandée (comme la distance par défaut de Chroma).
    """
    def emb(texte):
        if texte == requete:
            return [1.0, 0.0]
        if texte not in table:
            raise KeyError(f"texte inconnu de l'embedder de test : {texte!r}")
        c = 1.0 - table[texte] / 2.0
        return [c, math.sqrt(max(0.0, 1.0 - c * c))]
    return emb


class FausseCollection:
    def __init__(self, nom="faux", embedder=None):
        self.nom = nom
        self.embedder = embedder or embedder_par_defaut
        self.rows = {}                      # id -> (document, métadonnées)
        self.appels = collections.Counter()
        self.echec_sur = set()              # noms de méthodes qui lèvent (simulation de panne)

    def _appel(self, nom):
        self.appels[nom] += 1
        if nom in self.echec_sur:
            raise RuntimeError(f"panne simulée : {nom}")

    @staticmethod
    def _verifier_meta(meta):
        if meta is None:
            return None
        if not isinstance(meta, dict) or not meta:
            raise ValueError("Expected metadata to be a non-empty dict")
        for cle, val in meta.items():
            if not isinstance(val, (str, int, float, bool)):
                raise ValueError(f"Expected metadata value for {cle!r} to be str, int, float or bool, got {type(val)}")
        return dict(meta)

    def inserer_brut(self, doc_id, document, meta):
        """Insère SANS validation : simule des données héritées (métadonnées absentes ou abîmées)."""
        self.rows[doc_id] = (document, meta)

    def count(self):
        self._appel("count")
        return len(self.rows)

    def add(self, documents, metadatas=None, ids=None):
        self._appel("add")
        if not documents:
            raise ValueError("Expected non-empty documents")
        if ids is None or len(ids) != len(documents):
            raise ValueError("ids et documents de longueurs différentes")
        metadatas = metadatas if metadatas is not None else [None] * len(documents)
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            if doc_id in self.rows:
                raise ValueError(f"Expected IDs to be unique, found duplicate: {doc_id}")
            if not isinstance(doc, str):
                raise ValueError("document non textuel")
            self.rows[doc_id] = (doc, self._verifier_meta(meta))

    def get(self, ids=None):
        self._appel("get")
        cles = list(self.rows) if ids is None else [i for i in ids if i in self.rows]
        return {"ids": cles, "documents": [self.rows[k][0] for k in cles],
                "metadatas": [copy.deepcopy(self.rows[k][1]) for k in cles]}

    def update(self, ids, metadatas=None, documents=None):
        self._appel("update")
        for i, doc_id in enumerate(ids):
            if doc_id not in self.rows:
                raise ValueError(f"ID introuvable pour la mise à jour : {doc_id}")
            doc, meta = self.rows[doc_id]
            self.rows[doc_id] = (documents[i] if documents else doc,
                                 self._verifier_meta(metadatas[i]) if metadatas else meta)

    def delete(self, ids):
        self._appel("delete")
        for doc_id in ids:
            self.rows.pop(doc_id, None)     # comme Chroma : supprimer un id absent est silencieux

    def query(self, query_texts, n_results=10):
        self._appel("query")
        q = self.embedder(query_texts[0])
        scores = []
        for doc_id, (doc, _) in self.rows.items():
            e = self.embedder(doc)
            scores.append((sum((a - b) ** 2 for a, b in zip(q, e)), doc_id))
        scores.sort()
        scores = scores[:n_results]
        return {"ids": [[i for _, i in scores]], "documents": [[self.rows[i][0] for _, i in scores]],
                "metadatas": [[copy.deepcopy(self.rows[i][1]) for _, i in scores]],
                "distances": [[d for d, _ in scores]]}

    def cloner(self):
        copie = FausseCollection(self.nom, self.embedder)
        copie.rows = copy.deepcopy(self.rows)
        return copie


class Horloge:
    """Horloge contrôlable : les tests de decay n'attendent jamais le temps réel."""

    def __init__(self, debut):
        self.t = debut

    def __call__(self):
        return self.t

    def avancer(self, heures=0.0, jours=0.0):
        self.t += timedelta(hours=heures, days=jours)


def installer_faux_modules(monkeypatch, embedder=None):
    """Remplace `chromadb` et `ollama` pour importer l'ancien main.py sans dépendances réelles."""
    magasins = {}

    class PersistentClient:
        def __init__(self, path):
            self.collections = magasins.setdefault(path, {})

        def get_or_create_collection(self, name):
            return self.collections.setdefault(name, FausseCollection(name, embedder))

    chroma = types.ModuleType("chromadb")
    chroma.PersistentClient = PersistentClient
    chroma.__version__ = "faux"
    ollama = types.ModuleType("ollama")

    def pas_d_ollama(*a, **k):
        raise RuntimeError("pas d'Ollama dans ce test")
    ollama.chat = pas_d_ollama
    ollama.list = pas_d_ollama
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    monkeypatch.setitem(sys.modules, "ollama", ollama)
    return magasins

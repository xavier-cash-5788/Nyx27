"""
nyx/embeddings.py : fonction d'embedding pour ChromaDB via Ollama (module 2b/9).

Rôle : fournir à ChromaDB une fonction `__call__(input: List[str]) -> List[List[float]]`
qui utilise le modèle `nomic-embed-text` servi par Ollama.

Points clés :
  - Nomic exige des PRÉFIXES de tâche pour des résultats optimaux :
      * indexation  : "search_document: " + texte
      * requête     : "search_query: "    + texte
    Sans ces préfixes, la qualité de recherche chute nettement.
  - Le modèle est interrogé via l'API HTTP d'Ollama (/api/embed), pas via le client
    ollama Python (qui ne gère pas le batch aussi proprement selon les versions).
  - Un cache LRU évite de recalculer l'embedding d'un texte déjà vu (utile quand
    ChromaDB réindexe ou quand on migre).
  - En cas d'erreur (Ollama down), on lève une exception claire : Nyx doit savoir
    qu'elle ne peut pas indexer/requêter, pas silencieusement retourner du bruit.

Compatibilité ChromaDB :
  - ChromaDB accepte une `EmbeddingFunction` qui implémente `__call__`.
  - Elle est utilisée automatiquement par `collection.add()` et `collection.query()`
    si on la passe au moment de la création de la collection.
  - IMPORTANT : ChromaDB exige la même fonction à la création ET à l'ouverture,
    sinon il tente de re-télécharger un modèle ou de re-embedder.
"""
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import List, Optional

_log = logging.getLogger("nyx.embeddings")

# Dimensions connues de nomic-embed-text (v1.5)
NOMIC_DIM = 768
NOMIC_MODELE_DEFAUT = "nomic-embed-text:latest"
OLLAMA_URL_DEFAUT = "http://localhost:11434"

# Préfixes de tâche exigés par nomic-embed-text
PREFIXE_DOCUMENT = "search_document: "
PREFIXE_REQUETE = "search_query: "

# Taille du cache LRU (nombre de textes dont on garde l'embedding)
TAILLE_CACHE_DEFAUT = 2048

# Timeout par défaut pour un appel HTTP à Ollama (secondes)
TIMEOUT_DEFAUT = 60.0


class ErreurEmbedding(Exception):
    """Erreur générique lors du calcul d'un embedding."""


class OllamaIndisponible(ErreurEmbedding):
    """Ollama n'est pas joignable."""


class ModeleEmbeddingIntrouvable(ErreurEmbedding):
    """Le modèle d'embedding n'est pas installé dans Ollama."""


class OllamaEmbeddingFunction:
    """
    Fonction d'embedding compatible ChromaDB, qui interroge Ollama.

    Utilisation typique :
        ef = OllamaEmbeddingFunction(model="nomic-embed-text:latest")
        collection = client.get_or_create_collection(name="souvenirs", embedding_function=ef)
        collection.add(documents=["..."], ...)   # appelle ef(["..."], type="document")
        collection.query(query_texts=["..."])    # appelle ef(["..."], type="requête")

    Note : ChromaDB appelle `__call__(input)` SANS préciser s'il s'agit d'un
    document ou d'une requête. On ne peut donc pas distinguer les deux cas
    depuis la signature. Solution : exposer deux méthodes explicites
    (`embed_documents`, `embed_query`) ET une `__call__` par défaut qui
    utilise le préfixe document (le plus fréquent : l'ajout). Les appels de
    requête devront passer par une enveloppe (voir `ChromaNomicEF` ci-dessous)
    ou par une sous-classe de Chroma qui force le préfixe requête.
    """

    def __init__(
        self,
        model: str = NOMIC_MODELE_DEFAUT,
        base_url: str = OLLAMA_URL_DEFAUT,
        timeout: float = TIMEOUT_DEFAUT,
        taille_cache: int = TAILLE_CACHE_DEFAUT,
        dimension: int = NOMIC_DIM,
        prefixe_document: str = PREFIXE_DOCUMENT,
        prefixe_requete: str = PREFIXE_REQUETE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.dimension = int(dimension)
        self.prefixe_document = prefixe_document
        self.prefixe_requete = prefixe_requete

        # Cache LRU thread-safe : clé = (préfixe, texte), valeur = vecteur
        self._cache = OrderedDict()
        self._taille_cache = max(0, int(taille_cache))
        self._verrou_cache = threading.RLock()

        # Statistiques (utiles pour le debug et le check)
        self.stats = {
            "appels": 0,
            "textes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "erreurs": 0,
            "duree_totale_s": 0.0,
        }

    # ---------------------------------------------------------------- vérification

    def verifier(self, timeout: Optional[float] = None) -> dict:
        """
        Vérifie qu'Ollama répond et que le modèle est installé.
        Retourne {"joignable", "modele_installe", "modeles_disponibles", "message"}.
        Ne lève jamais.
        """
        delai = timeout if timeout is not None else self.timeout
        try:
            modeles = self._lister_modeles(delai)
        except OllamaIndisponible as e:
            return {
                "joignable": False,
                "modele_installe": False,
                "modeles_disponibles": [],
                "message": str(e),
            }
        installe = any(m == self.model or (":" not in self.model and m == f"{self.model}:latest")
                       for m in modeles)
        message = (
            f"Modèle '{self.model}' prêt."
            if installe
            else f"Modèle '{self.model}' introuvable. Installez-le :  ollama pull {self.model}"
        )
        return {
            "joignable": True,
            "modele_installe": installe,
            "modeles_disponibles": modeles,
            "message": message,
        }

    def _lister_modeles(self, timeout: float) -> List[str]:
        """Liste les modèles installés dans Ollama. Lève OllamaIndisponible si l'API ne répond pas."""
        try:
            reponse = self._requete_http(
                "GET", f"{self.base_url}/api/tags", corps=None, timeout=timeout
            )
        except urllib.error.URLError as e:
            raise OllamaIndisponible(
                f"Ollama est injoignable sur {self.base_url}. Est-il lancé ? (commande : ollama serve). Détail : {e}"
            ) from e
        modeles = []
        for mod in reponse.get("models", []) or []:
            if isinstance(mod, dict):
                nom = mod.get("name") or mod.get("model")
                if nom:
                    modeles.append(nom)
        return modeles

    # ---------------------------------------------------------------- embedding

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Signature attendue par ChromaDB. Par défaut, préfixe 'search_document'
        (cas le plus fréquent : indexation).
        Pour une requête, utiliser `embed_query()` explicitement.
        """
        return self.embed_documents(input)

    def embed_documents(self, textes: List[str]) -> List[List[float]]:
        """Encode une liste de textes comme documents indexables."""
        return self._embed(textes, prefixe=self.prefixe_document)

    def embed_query(self, texte: str) -> List[float]:
        """Encode un texte comme requête de recherche."""
        return self._embed([texte], prefixe=self.prefixe_requete)[0]

    def embed_requetes(self, textes: List[str]) -> List[List[float]]:
        """Encode une liste de textes comme requêtes de recherche."""
        return self._embed(textes, prefixe=self.prefixe_requete)

    # ---------------------------------------------------------------- interne

    def _cle_cache(self, prefixe: str, texte: str) -> str:
        h = hashlib.sha1()
        h.update(prefixe.encode("utf-8"))
        h.update(b"\x00")
        h.update(texte.encode("utf-8"))
        return h.hexdigest()

    def _lire_cache(self, cle: str) -> Optional[List[float]]:
        if self._taille_cache == 0:
            return None
        with self._verrou_cache:
            vecteur = self._cache.get(cle)
            if vecteur is not None:
                self._cache.move_to_end(cle)  # LRU
            return vecteur

    def _ecrire_cache(self, cle: str, vecteur: List[float]) -> None:
        if self._taille_cache == 0:
            return
        with self._verrou_cache:
            self._cache[cle] = vecteur
            self._cache.move_to_end(cle)
            while len(self._cache) > self._taille_cache:
                self._cache.popitem(last=False)

    def _embed(self, textes: List[str], prefixe: str) -> List[List[float]]:
        """
        Cœur du calcul : gère cache, batch, préfixes, erreurs.
        Lève OllamaIndisponible ou ModeleEmbeddingIntrouvable selon le cas.
        """
        if not isinstance(textes, (list, tuple)):
            textes = [textes]
        textes = [t if isinstance(t, str) else str(t) for t in textes]
        if not textes:
            return []

        self.stats["appels"] += 1
        self.stats["textes"] += len(textes)
        debut = time.monotonic()

        # 1. Séparer ce qui est en cache de ce qu'il faut calculer
        resultats: List[Optional[List[float]]] = [None] * len(textes)
        a_calculer_textes: List[str] = []
        a_calculer_indices: List[int] = []
        for i, texte in enumerate(textes):
            cle = self._cle_cache(prefixe, texte)
            vecteur = self._lire_cache(cle)
            if vecteur is not None:
                resultats[i] = vecteur
                self.stats["cache_hits"] += 1
            else:
                a_calculer_textes.append(prefixe + texte)
                a_calculer_indices.append(i)
                self.stats["cache_misses"] += 1

        # 2. Calculer ce qui manque en un seul appel batch
        if a_calculer_textes:
            vecteurs = self._appel_ollama(a_calculer_textes)
            if len(vecteurs) != len(a_calculer_textes):
                raise ErreurEmbedding(
                    f"Ollama a renvoyé {len(vecteurs)} vecteurs pour "
                    f"{len(a_calculer_textes)} textes envoyés."
                )
            for idx, vecteur in zip(a_calculer_indices, vecteurs):
                resultats[idx] = vecteur
                cle = self._cle_cache(prefixe, textes[idx])
                self._ecrire_cache(cle, vecteur)

        self.stats["duree_totale_s"] += time.monotonic() - debut
        return [v if v is not None else [] for v in resultats]

    def _appel_ollama(self, textes_prefixes: List[str]) -> List[List[float]]:
        """Interroge /api/embed d'Ollama avec un batch de textes préfixés."""
        corps = {"model": self.model, "input": textes_prefixes}
        try:
            reponse = self._requete_http(
                "POST", f"{self.base_url}/api/embed", corps=corps, timeout=self.timeout
            )
        except urllib.error.HTTPError as e:
            self.stats["erreurs"] += 1
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            if e.code == 404:
                raise ModeleEmbeddingIntrouvable(
                    f"Modèle d'embedding '{self.model}' introuvable dans Ollama. "
                    f"Installez-le :  ollama pull {self.model}. Détail : {detail}"
                ) from e
            raise ErreurEmbedding(f"Ollama a répondu {e.code} : {detail}") from e
        except urllib.error.URLError as e:
            self.stats["erreurs"] += 1
            raise OllamaIndisponible(
                f"Ollama est injoignable sur {self.base_url}. Est-il lancé ? "
                f"(commande : ollama serve). Détail : {e}"
            ) from e
        except Exception as e:  # noqa: BLE001
            self.stats["erreurs"] += 1
            raise ErreurEmbedding(f"Erreur inattendue lors de l'appel Ollama : {e}") from e

        embeddings = reponse.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(textes_prefixes):
            raise ErreurEmbedding(
                f"Réponse Ollama inattendue : 'embeddings' absent ou de taille incorrecte. "
                f"Reçu : {list(reponse.keys())}"
            )
        # Vérification de la dimension
        for v in embeddings:
            if not isinstance(v, list) or len(v) != self.dimension:
                raise ErreurEmbedding(
                    f"Vecteur de dimension {len(v) if isinstance(v, list) else '?'} "
                    f"au lieu de {self.dimension} attendu pour {self.model}."
                )
        return embeddings

    @staticmethod
    def _requete_http(methode: str, url: str, corps: Optional[dict], timeout: float) -> dict:
        """Requête HTTP minimale vers Ollama, sans dépendance externe (urllib)."""
        data = None
        en_tetes = {"Accept": "application/json"}
        if corps is not None:
            data = json.dumps(corps).encode("utf-8")
            en_tetes["Content-Type"] = "application/json"
        requete = urllib.request.Request(url, data=data, headers=en_tetes, method=methode)
        with urllib.request.urlopen(requete, timeout=timeout) as reponse:
            brut = reponse.read().decode("utf-8")
        try:
            return json.loads(brut)
        except ValueError as e:
            raise ErreurEmbedding(f"Réponse Ollama illisible (JSON invalide) : {brut[:200]}") from e

    # ---------------------------------------------------------------- utilitaires

    def vider_cache(self) -> None:
        with self._verrou_cache:
            self._cache.clear()

    def resume_stats(self) -> str:
        s = self.stats
        taux = (s["cache_hits"] / s["cache_misses"]) if s["cache_misses"] else float("inf")
        return (
            f"Embedding '{self.model}' : {s['appels']} appel(s), {s['textes']} texte(s), "
            f"{s['cache_hits']} hit(s), {s['cache_misses']} miss, "
            f"{s['erreurs']} erreur(s), {s['duree_totale_s']:.1f}s au total "
            f"(ratio hit/miss = {taux:.2f})"
        )


class ChromaNomicEF(OllamaEmbeddingFunction):
    """
    Variante qui distingue les requêtes des documents, pour ChromaDB.

    ChromaDB appelle `__call__(input)` sans préciser la nature (document ou
    requête). Pour appliquer le bon préfixe Nomic, on utilise une astuce :
    `Chroma.query()` appelle `embedding_function(input)`, tandis que
    `Chroma.add()` fait de même. Impossible de distinguer sans patcher Chroma.

    Solution retenue : NE PAS utiliser cette classe directement avec Chroma.
    À la place, dans `memoire.py` et `journal.py`, on appelle explicitement
    `embed_query()` ou `embed_documents()` AVANT Chroma, et on passe les
    vecteurs précalculés à Chroma via `embeddings=[...]` au lieu de
    `query_texts=[...]` ou `documents=[...]`.

    Cette classe existe donc surtout comme marqueur/documentation, et pour
    fournir `__call__` (fallback document) si un jour Chroma l'appelle.
    """
    pass


def creer_embedding_function(
    model: str = NOMIC_MODELE_DEFAUT,
    base_url: str = OLLAMA_URL_DEFAUT,
    timeout: float = TIMEOUT_DEFAUT,
    taille_cache: int = TAILLE_CACHE_DEFAUT,
    dimension: int = NOMIC_DIM,
) -> OllamaEmbeddingFunction:
    """Fabrique standard utilisée par l'agent."""
    return OllamaEmbeddingFunction(
        model=model, base_url=base_url, timeout=timeout,
        taille_cache=taille_cache, dimension=dimension,
    )


# ---------------------------------------------------------------- test rapide

if __name__ == "__main__":
    # Petit test manuel : envoie deux textes à Ollama et vérifie la dimension.
    import sys
    logging.basicConfig(level=logging.INFO)
    ef = OllamaEmbeddingFunction()
    diag = ef.verifier()
    print("Vérification :", diag["message"])
    if not diag["joignable"] or not diag["modele_installe"]:
        sys.exit(1)
    v1 = ef.embed_query("Qu'est-ce que la philosophie antique ?")
    v2 = ef.embed_documents(["J'aime la philosophie antique.", "Mon chien s'appelle Rex."])
    print(f"Dimension requête   : {len(v1)}")
    print(f"Dimension documents : {[len(v) for v in v2]}")
    print(ef.resume_stats())
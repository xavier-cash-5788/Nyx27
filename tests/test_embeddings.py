"""
Tests unitaires pour nyx/embeddings.py.

Stratégie : un petit serveur HTTP local (ThreadingHTTPServer) simule l'API Ollama
pour les routes /api/tags et /api/embed. Aucun appel réseau externe n'est effectué.

Cas couverts :
  - construction (défauts, paramètres personnalisés)
  - application correcte des préfixes Nomic (search_document / search_query)
  - cache LRU (hits, misses, taille, thread-safety)
  - batch (un seul appel HTTP pour N textes)
  - gestion d'erreurs : Ollama down, 404, réponse malformée, dimension incorrecte
  - verifier() sans lever
  - compatibilité ChromaDB : __call__(input) retourne bien List[List[float]]
"""
import json
import math
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nyx.embeddings import (
    NOMIC_DIM,
    NOMIC_MODELE_DEFAUT,
    OLLAMA_URL_DEFAUT,
    PREFIXE_DOCUMENT,
    PREFIXE_REQUETE,
    ErreurEmbedding,
    ModeleEmbeddingIntrouvable,
    OllamaEmbeddingFunction,
    OllamaIndisponible,
    creer_embedding_function,
)


# ================================================================ serveur de test

class FauxServeurOllama(ThreadingHTTPServer):
    """Serveur HTTP local qui imite l'API Ollama pour /api/tags et /api/embed."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Gestionnaire)
        self.mode = "normal"
        self.modeles = ["nomic-embed-text:latest", "qwen2.5:7b"]
        self.dimension = NOMIC_DIM
        self.texte_recus = []         # [(model, input)]
        self.delai = 0.0              # délai avant de répondre
        self.verrou = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def arreter(self):
        self.shutdown()
        self.server_close()


class Gestionnaire(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence

    def _json(self, code, objet):
        corps = json.dumps(objet).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self):
        s = self.server
        if s.delai:
            time.sleep(s.delai)
        if self.path == "/api/tags":
            if s.mode == "down":
                return self._json(500, {"error": "serveur simulé down"})
            return self._json(200, {"models": [{"name": n, "model": n} for n in s.modeles]})
        self._json(404, {"error": "route inconnue"})

    def do_POST(self):
        s = self.server
        if s.delai:
            time.sleep(s.delai)
        if self.path != "/api/embed":
            return self._json(404, {"error": "route inconnue"})
        try:
            longueur = int(self.headers.get("Content-Length", 0))
            corps = json.loads(self.rfile.read(longueur) or b"{}")
        except (ValueError, TypeError):
            return self._json(400, {"error": "corps JSON invalide"})

        model = corps.get("model")
        entrees = corps.get("input") or []
        if not isinstance(entrees, list):
            entrees = [entrees]

        with s.verrou:
            s.texte_recus.append((model, list(entrees)))

        if s.mode == "404":
            return self._json(404, {"error": f"model '{model}' not found"})
        if s.mode == "dim-incorrecte":
            return self._json(200, {"embeddings": [[0.0] * 10 for _ in entrees]})
        if s.mode == "sans-embeddings":
            return self._json(200, {"model": model})
        if s.mode == "erreur-500":
            return self._json(500, {"error": "erreur interne simulée"})

        # Mode normal : on renvoie des vecteurs déterministes basés sur le hash du texte
        dimension = s.dimension
        embeddings = []
        for texte in entrees:
            # Vecteur déterministe : chaque texte a son vecteur unique, dimension correcte
            base = sum(ord(c) for c in str(texte))
            v = [math.sin((base + i) * 0.001) for i in range(dimension)]
            # Normalise (comme le fait nomic)
            norme = math.sqrt(sum(x * x for x in v)) or 1.0
            embeddings.append([x / norme for x in v])
        self._json(200, {"model": model, "embeddings": embeddings})


@pytest.fixture
def serveur():
    s = FauxServeurOllama()
    yield s
    s.arreter()


def ef(serveur, **kw):
    """Fabrique une OllamaEmbeddingFunction branchée sur le serveur de test."""
    return OllamaEmbeddingFunction(base_url=serveur.url, **kw)


# ================================================================ construction

def test_parametres_par_defaut():
    e = OllamaEmbeddingFunction()
    assert e.model == NOMIC_MODELE_DEFAUT
    assert e.base_url == OLLAMA_URL_DEFAUT
    assert e.dimension == NOMIC_DIM
    assert e.prefixe_document == PREFIXE_DOCUMENT
    assert e.prefixe_requete == PREFIXE_REQUETE


def test_parametres_personnalises(serveur):
    e = OllamaEmbeddingFunction(
        model="autre:1b", base_url=serveur.url + "/",
        timeout=5.0, taille_cache=10, dimension=128,
        prefixe_document="doc: ", prefixe_requete="qry: ",
    )
    assert e.model == "autre:1b"
    assert e.base_url == serveur.url  # slash final retiré
    assert e.timeout == 5.0
    assert e.dimension == 128
    assert e.prefixe_document == "doc: "
    assert e.prefixe_requete == "qry: "


def test_fabrique():
    e = creer_embedding_function(model="x", base_url="http://x")
    assert isinstance(e, OllamaEmbeddingFunction)
    assert e.model == "x" and e.base_url == "http://x"


# ================================================================ vérification

def test_verifier_modele_installe(serveur):
    diag = ef(serveur).verifier()
    assert diag["joignable"] is True
    assert diag["modele_installe"] is True
    assert "nomic-embed-text" in diag["message"]
    assert "prêt" in diag["message"]


def test_verifier_modele_absent(serveur):
    serveur.modeles = ["qwen2.5:7b"]
    diag = ef(serveur).verifier()
    assert diag["joignable"] is True
    assert diag["modele_installe"] is False
    assert "ollama pull" in diag["message"]


def test_verifier_modele_avec_latest_implicite(serveur):
    serveur.modeles = ["nomic-embed-text:latest"]
    diag = ef(serveur, model="nomic-embed-text").verifier()
    assert diag["modele_installe"] is True


def test_verifier_ollama_down():
    # Adresse invalide : port fermé
    e = OllamaEmbeddingFunction(base_url="http://127.0.0.1:9", timeout=1.0)
    diag = e.verifier()
    assert diag["joignable"] is False
    assert diag["modele_installe"] is False
    assert "ollama serve" in diag["message"]


# ================================================================ préfixes Nomic

def test_embed_documents_applique_le_prefixe_document(serveur):
    e = ef(serveur)
    e.embed_documents(["texte 1", "texte 2"])
    model, entrees = serveur.texte_recus[-1]
    assert entrees == ["search_document: texte 1", "search_document: texte 2"]


def test_embed_query_applique_le_prefixe_requete(serveur):
    e = ef(serveur)
    e.embed_query("ma requête")
    model, entrees = serveur.texte_recus[-1]
    assert entrees == ["search_query: ma requête"]


def test_embed_requetes_applique_le_prefixe_requete(serveur):
    e = ef(serveur)
    e.embed_requetes(["r1", "r2"])
    model, entrees = serveur.texte_recus[-1]
    assert entrees == ["search_query: r1", "search_query: r2"]


def test_call_utilise_prefixe_document(serveur):
    """__call__ est appelé par ChromaDB pour l'indexation → préfixe document."""
    e = ef(serveur)
    vecteurs = e(["a", "b"])
    model, entrees = serveur.texte_recus[-1]
    assert entrees == ["search_document: a", "search_document: b"]
    assert len(vecteurs) == 2
    assert all(len(v) == NOMIC_DIM for v in vecteurs)


def test_prefixes_personnalises(serveur):
    e = ef(serveur, prefixe_document="DOC> ", prefixe_requete="QRY> ")
    e.embed_documents(["x"])
    e.embed_query("y")
    assert serveur.texte_recus[-2][1] == ["DOC> x"]
    assert serveur.texte_recus[-1][1] == ["QRY> y"]


# ================================================================ retour de valeurs

def test_embed_documents_retourne_bon_nombre_de_vecteurs(serveur):
    e = ef(serveur)
    vecteurs = e.embed_documents(["a", "b", "c"])
    assert len(vecteurs) == 3
    assert all(isinstance(v, list) for v in vecteurs)
    assert all(len(v) == NOMIC_DIM for v in vecteurs)
    assert all(all(isinstance(x, float) for x in v) for v in vecteurs)


def test_embed_query_retourne_un_seul_vecteur(serveur):
    e = ef(serveur)
    v = e.embed_query("test")
    assert isinstance(v, list)
    assert len(v) == NOMIC_DIM


def test_embed_sur_liste_vide(serveur):
    e = ef(serveur)
    assert e.embed_documents([]) == []
    assert e.embed_requetes([]) == []


def test_embed_accepte_une_chaine_au_lieu_d_une_liste(serveur):
    """Robustesse : si on passe une string au lieu d'une liste, on l'emballe."""
    e = ef(serveur)
    vecteurs = e.embed_documents("texte seul")
    assert len(vecteurs) == 1
    assert serveur.texte_recus[-1][1] == ["search_document: texte seul"]


def test_embed_convertit_les_non_strings(serveur):
    e = ef(serveur)
    e.embed_documents([42, None, {"x": 1}])
    model, entrees = serveur.texte_recus[-1]
    assert len(entrees) == 3
    assert all(isinstance(x, str) for x in entrees)


# ================================================================ cache LRU

def test_cache_hit_evite_un_appel_reseau(serveur):
    e = ef(serveur, taille_cache=10)
    e.embed_query("bonjour")
    appels_avant = len(serveur.texte_recus)
    e.embed_query("bonjour")  # doit être servi par le cache
    assert len(serveur.texte_recus) == appels_avant
    assert e.stats["cache_hits"] == 1
    assert e.stats["cache_misses"] == 1


def test_cache_distingue_document_et_requete(serveur):
    """Le même texte indexé et requêté NE DOIT PAS partager le cache (préfixes différents)."""
    e = ef(serveur)
    e.embed_documents(["bonjour"])
    e.embed_query("bonjour")
    assert len(serveur.texte_recus) == 2  # deux appels distincts
    assert e.stats["cache_hits"] == 0
    assert e.stats["cache_misses"] == 2


def test_cache_evince_les_plus_anciens(serveur):
    e = ef(serveur, taille_cache=2)
    e.embed_documents(["a"])
    e.embed_documents(["b"])
    e.embed_documents(["c"])  # évince "a"
    e.embed_documents(["a"])  # doit être un miss (a été évincé)
    e.embed_documents(["c"])  # doit être un hit (c est toujours en cache)
    assert e.stats["cache_hits"] == 1
    assert e.stats["cache_misses"] == 4


def test_cache_desactive(serveur):
    e = ef(serveur, taille_cache=0)
    e.embed_documents(["a"])
    e.embed_documents(["a"])
    e.embed_documents(["a"])
    assert e.stats["cache_hits"] == 0
    assert e.stats["cache_misses"] == 3
    assert len(serveur.texte_recus) == 3  # trois appels distincts


def test_vider_cache(serveur):
    e = ef(serveur)
    e.embed_documents(["a"])
    assert len(e._cache) == 1
    e.vider_cache()
    assert len(e._cache) == 0


def test_cache_melange_hits_et_misses(serveur):
    e = ef(serveur)
    e.embed_documents(["a", "b"])
    appels_avant = len(serveur.texte_recus)
    # "a" est en cache, "c" non : un seul appel réseau pour "c"
    e.embed_documents(["a", "c"])
    assert len(serveur.texte_recus) == appels_avant + 1
    model, entrees = serveur.texte_recus[-1]
    assert entrees == ["search_document: c"]


# ================================================================ erreurs

def test_ollama_down_leve_ollamaindisponible():
    e = OllamaEmbeddingFunction(base_url="http://127.0.0.1:9", timeout=1.0)
    with pytest.raises(OllamaIndisponible) as info:
        e.embed_query("test")
    assert "ollama serve" in str(info.value)
    assert e.stats["erreurs"] == 1


def test_modele_absent_leve_modeleintrouvable(serveur):
    serveur.mode = "404"
    e = ef(serveur)
    with pytest.raises(ModeleEmbeddingIntrouvable) as info:
        e.embed_query("test")
    assert "ollama pull" in str(info.value)
    assert e.stats["erreurs"] == 1


def test_reponse_sans_embeddings_leve_erreur(serveur):
    serveur.mode = "sans-embeddings"
    e = ef(serveur)
    with pytest.raises(ErreurEmbedding) as info:
        e.embed_query("test")
    assert "embeddings" in str(info.value).lower()


def test_dimension_incorrecte_leve_erreur(serveur):
    serveur.mode = "dim-incorrecte"
    e = ef(serveur)  # attend 768 dims
    with pytest.raises(ErreurEmbedding) as info:
        e.embed_query("test")
    assert "dimension" in str(info.value).lower()


def test_erreur_500_leve_erreurembedding(serveur):
    serveur.mode = "erreur-500"
    e = ef(serveur)
    with pytest.raises(ErreurEmbedding) as info:
        e.embed_query("test")
    assert "500" in str(info.value)


def test_le_cache_n_est_pas_empoisonne_par_une_erreur(serveur):
    """Après une erreur, un appel réussi doit fonctionner (pas de cache vide toxique)."""
    e = ef(serveur)
    serveur.mode = "404"
    with pytest.raises(ModeleEmbeddingIntrouvable):
        e.embed_query("test")
    serveur.mode = "normal"
    v = e.embed_query("test")
    assert len(v) == NOMIC_DIM


# ================================================================ robustesse

def test_timeout_est_respecte(serveur):
    serveur.delai = 2.0
    e = ef(serveur, timeout=0.2)
    debut = time.monotonic()
    with pytest.raises((OllamaIndisponible, ErreurEmbedding)):
        e.embed_query("test")
    assert time.monotonic() - debut < 1.5


def test_appels_concurrents_ne_corrompent_pas_le_cache(serveur):
    e = ef(serveur, taille_cache=50)
    erreurs = []

    def travailleur(n):
        try:
            for i in range(20):
                e.embed_documents([f"thread-{n}-{i % 5}"])
                e.embed_query(f"requete-{n}-{i % 3}")
        except Exception as exc:  # noqa: BLE001
            erreurs.append(exc)

    threads = [threading.Thread(target=travailleur, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not erreurs
    # Le cache ne doit pas dépasser la taille max
    assert len(e._cache) <= 50


def test_batch_reduit_le_nombre_d_appels(serveur):
    """Un batch de 10 textes ne fait qu'un seul appel HTTP."""
    e = ef(serveur)
    e.embed_documents([f"texte {i}" for i in range(10)])
    assert len(serveur.texte_recus) == 1
    model, entrees = serveur.texte_recus[-1]
    assert len(entrees) == 10


# ================================================================ statistiques

def test_stats_comptent_correctement(serveur):
    e = ef(serveur)
    e.embed_documents(["a", "b"])   # 1 appel, 2 textes, 2 misses
    e.embed_query("c")               # 1 appel, 1 texte, 1 miss
    e.embed_documents(["a"])         # 0 appel, 1 hit
    assert e.stats["appels"] == 3
    assert e.stats["textes"] == 4
    assert e.stats["cache_hits"] == 1
    assert e.stats["cache_misses"] == 3
    assert e.stats["erreurs"] == 0


def test_resume_stats_est_lisible(serveur):
    e = ef(serveur)
    e.embed_documents(["a"])
    e.embed_documents(["a"])  # hit
    resume = e.resume_stats()
    assert "nomic-embed-text" in resume
    assert "cache_hits" in resume or "hit" in resume.lower()


# ================================================================ compatibilité ChromaDB

def test_signature_chroma():
    """ChromaDB appelle embedding_function(input) avec une liste, attend une liste de vecteurs."""
    e = OllamaEmbeddingFunction(base_url="http://127.0.0.1:9")  # jamais appelé
    # Vérifie que __call__ accepte une liste et retourne une liste
    # (on mocke _appel_ollama pour éviter un vrai appel réseau)
    e._appel_ollama = lambda textes: [[0.0] * NOMIC_DIM for _ in textes]
    resultat = e(["texte 1", "texte 2"])
    assert isinstance(resultat, list)
    assert len(resultat) == 2
    assert all(isinstance(v, list) and len(v) == NOMIC_DIM for v in resultat)
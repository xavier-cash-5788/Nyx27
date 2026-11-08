"""
Intégration : la vraie bibliothèque `ollama` (client HTTP réel) contre un petit serveur qui imite l'API
d'Ollama (/api/chat en flux NDJSON, /api/tags). Prouve, sans avoir besoin d'Ollama, que :
  - le flux réel est bien lu et assemblé,
  - annuler ou dépasser le délai FERME la connexion (le serveur le détecte : la génération s'arrêterait),
  - une erreur 404 / une connexion refusée sont correctement traduites.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nyx.config import ConfigNyx
from nyx.llm import ClientLLM, LLMAnnule, LLMIndisponible, LLMTimeout, ModeleIntrouvable

ollama = pytest.importorskip("ollama")
MESSAGES = [{"role": "user", "content": "Bonjour"}]


class ServeurOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Gestionnaire)
        self.mode = "normal"
        self.morceaux = ["Bon", "jour", " Xavier"]
        self.pause = 0.0
        self.avant = 0.0
        self.deconnecte = threading.Event()
        self.recus = []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Gestionnaire(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, objet):
        corps = json.dumps(objet).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]})
        else:
            self._json(404, {"error": "inconnu"})

    def do_POST(self):
        s = self.server
        requete = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        s.recus.append(requete)
        if s.mode == "404":
            return self._json(404, {"error": f"model '{requete.get('model')}' not found"})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            time.sleep(s.avant)
            for t in s.morceaux:
                ligne = {"model": requete.get("model"), "created_at": "2026-01-01T00:00:00Z",
                         "message": {"role": "assistant", "content": t}, "done": False}
                self.wfile.write((json.dumps(ligne) + "\n").encode())
                self.wfile.flush()
                time.sleep(s.pause)
            fin = {"model": requete.get("model"), "created_at": "2026-01-01T00:00:00Z",
                   "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}
            self.wfile.write((json.dumps(fin) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            s.deconnecte.set()  # le client a fermé la connexion : Ollama arrêterait de générer


@pytest.fixture
def serveur():
    s = ServeurOllama()
    yield s
    s.shutdown()
    s.server_close()


def client(serveur, **kw):
    return ClientLLM(ConfigNyx.creer(TIMEOUT_LLM=10.0), backend=ollama.Client(host=serveur.url), **kw)


def test_flux_reel_assemble_et_parametres_recus_par_le_serveur(serveur):
    c = client(serveur)
    recus = []
    rep = c.chat(MESSAGES, temperature=0.1, format="json", sur_morceau=recus.append)
    assert rep.texte == "Bonjour Xavier" and recus == ["Bon", "jour", " Xavier"]
    r = serveur.recus[0]
    assert r["model"] == "qwen2.5:7b" and r["messages"] == MESSAGES and r["stream"] is True
    assert r["format"] == "json" and r["options"]["num_ctx"] == 4096 and r["options"]["temperature"] == 0.1


def test_annulation_ferme_vraiment_la_connexion(serveur):
    serveur.morceaux, serveur.pause = ["x"] * 300, 0.05
    c = client(serveur)
    stop = threading.Event()
    threading.Timer(0.25, stop.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMAnnule):
        c.chat(MESSAGES, annulation=stop)
    assert time.monotonic() - t0 < 1.0
    assert serveur.deconnecte.wait(3.0), "le serveur doit constater la déconnexion (génération arrêtée)"


def test_timeout_ferme_vraiment_la_connexion(serveur):
    serveur.morceaux, serveur.pause = ["x"] * 300, 0.05
    c = client(serveur)
    with pytest.raises(LLMTimeout):
        c.chat(MESSAGES, timeout=0.3)
    assert serveur.deconnecte.wait(3.0)


def test_timeout_avant_le_premier_octet(serveur):
    serveur.avant = 3.0
    c = client(serveur)
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        c.chat(MESSAGES, timeout=0.3)
    assert time.monotonic() - t0 < 1.0


def test_modele_introuvable_via_le_vrai_404(serveur):
    serveur.mode = "404"
    c = client(serveur, tentatives=3)
    with pytest.raises(ModeleIntrouvable) as info:
        c.chat(MESSAGES, modele="fantome:7b")
    assert "ollama pull fantome:7b" in str(info.value) and len(serveur.recus) == 1


def test_liste_des_modeles_et_verification_via_le_vrai_client(serveur):
    c = client(serveur)
    assert c.lister_modeles() == ["qwen2.5:7b"]
    v = c.verifier(["qwen2.5:7b", "absent:1b"])
    assert v["joignable"] and v["manquants"] == ["absent:1b"]


def test_serveur_arrete_donne_llmindisponible(serveur):
    c = client(serveur, tentatives=2, pause_tentative=0.01)
    serveur.shutdown()
    serveur.server_close()
    with pytest.raises(LLMIndisponible):
        c.chat(MESSAGES)

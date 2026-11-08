import threading
import time

import pytest

from nyx.config import ConfigNyx
from nyx.llm import (ClientLLM, ErreurLLM, LLMAnnule, LLMIndisponible, LLMTimeout, ModeleIntrouvable,
                     ReponseLLM, _importer_ollama, extraire_json)


# ---------------------------------------------------------------- faux Ollama

class FauxOllama:
    """Backend de test : enregistre les appels et rejoue des scénarios."""

    def __init__(self, scenario=None, liste=None):
        self.appels = []
        self.scenarios = [scenario] if callable(scenario) else list(scenario or [])
        self.liste = liste if liste is not None else {"models": [{"name": "qwen2.5:7b"}]}

    def chat(self, **kw):
        self.appels.append(kw)
        suivant = self.scenarios.pop(0) if len(self.scenarios) > 1 else self.scenarios[0]
        return suivant(kw)

    def list(self):
        if isinstance(self.liste, Exception):
            raise self.liste
        if callable(self.liste):
            return self.liste()
        return self.liste


def texte(*morceaux, delai=0.0, ferme=None, avant=0.0):
    """Scénario : un flux de morceaux. `ferme` est positionné quand le flux est fermé (arrêt côté serveur)."""
    def scenario(kw):
        def gen():
            try:
                if avant:
                    time.sleep(avant)
                for m in morceaux:
                    if delai:
                        time.sleep(delai)
                    yield {"message": {"role": "assistant", "content": m}, "done": False}
                yield {"message": {"role": "assistant", "content": ""}, "done": True}
            finally:
                if ferme is not None:
                    ferme.set()
        return gen()
    return scenario


def echec(erreur, *, pendant_le_flux=False):
    def scenario(kw):
        if not pendant_le_flux:
            raise erreur

        def gen():
            raise erreur
            yield  # pragma: no cover
        return gen()
    return scenario


@pytest.fixture
def cfg():
    return ConfigNyx.creer(TIMEOUT_LLM=5.0)


def client(cfg, scenario, **kw):
    backend = FauxOllama(scenario)
    return ClientLLM(cfg, backend=backend, pause_tentative=0.01, **kw), backend


MESSAGES = [{"role": "user", "content": "Bonjour"}]


# ---------------------------------------------------------------- réponse et paramètres

def test_assemble_les_morceaux_et_garde_la_forme_de_l_ancien_code(cfg):
    c, _ = client(cfg, texte("Bon", "jour", " Xavier"))
    rep = c.chat(MESSAGES)
    assert isinstance(rep, ReponseLLM)
    assert rep["message"]["content"] == "Bonjour Xavier" == rep.texte      # forme de l'ancien code
    assert rep["message"]["role"] == "assistant" and rep.morceaux == 3
    assert c.chat_texte(MESSAGES) == "Bonjour Xavier"


def test_parametres_transmis_au_backend(cfg):
    c, b = client(cfg, texte("ok"))
    c.chat(MESSAGES, format="json", temperature=0.2, max_tokens=50)
    kw = b.appels[0]
    assert kw["model"] == "qwen2.5:7b" and kw["messages"] == MESSAGES and kw["stream"] is True
    assert kw["format"] == "json"
    assert kw["options"] == {"num_ctx": 4096, "temperature": 0.2, "num_predict": 50}
    assert "think" not in kw


def test_les_options_absentes_ne_sont_pas_envoyees(cfg):
    c, b = client(cfg, texte("ok"))
    c.chat(MESSAGES)
    kw = b.appels[0]
    assert "format" not in kw and kw["options"] == {"num_ctx": 4096}


def test_penser_n_est_envoye_que_si_demande(cfg):
    c, b = client(cfg, texte("ok"))
    c.chat(MESSAGES, penser=False)
    assert b.appels[0]["think"] is False


def test_la_config_est_relue_a_chaque_appel(cfg):
    c, b = client(cfg, texte("ok"))
    c.chat(MESSAGES)
    cfg.definir("model_name", "gemma2:2b")
    cfg.definir("num_ctx", 2048)
    c.chat(MESSAGES)
    c.chat(MESSAGES, modele="autre:7b")
    assert [a["model"] for a in b.appels] == ["qwen2.5:7b", "gemma2:2b", "autre:7b"]
    assert b.appels[1]["options"]["num_ctx"] == 2048


def test_statistiques_du_dernier_appel(cfg):
    c, _ = client(cfg, texte("a", "bc", delai=0.02))
    c.chat(MESSAGES)
    d = c.dernier
    assert d["modele"] == "qwen2.5:7b" and d["caracteres"] == 3 and d["morceaux"] == 2
    assert 0 < d["premier_morceau_s"] <= d["duree_s"]


def test_formats_de_morceaux_dict_none_et_objets_reels(cfg):
    def scenario(kw):
        yield {"message": {"content": "A"}}
        yield {"message": {"content": None}}      # contenu absent
        yield {"pas_de_message": True}            # morceau inattendu
        yield {"message": {"content": "B"}}
    c, _ = client(cfg, scenario)
    assert c.chat(MESSAGES).texte == "AB"


def test_objets_chatresponse_de_la_vraie_bibliotheque(cfg):
    ollama = pytest.importorskip("ollama")

    def scenario(kw):
        for t in ("Sa", "lut"):
            yield ollama.ChatResponse(model="x", message=ollama.Message(role="assistant", content=t), done=False)
    c, _ = client(cfg, scenario)
    assert c.chat(MESSAGES).texte == "Salut"


# ---------------------------------------------------------------- streaming

def test_sur_morceau_recoit_les_deltas_dans_le_thread_appelant(cfg):
    c, _ = client(cfg, texte("un ", "deux ", "trois"))
    recus, threads = [], set()

    def cb(delta):
        recus.append(delta)
        threads.add(threading.current_thread())
    c.chat(MESSAGES, sur_morceau=cb)
    assert recus == ["un ", "deux ", "trois"]
    assert threads == {threading.current_thread()}


def test_un_callback_defaillant_ne_tue_pas_la_generation(cfg):
    c, _ = client(cfg, texte("a", "b", "c"))
    appels = []

    def cb(delta):
        appels.append(delta)
        raise RuntimeError("bug d'affichage")
    assert c.chat(MESSAGES, sur_morceau=cb).texte == "abc"
    assert appels == ["a"], "le callback est désactivé après sa première erreur"


# ---------------------------------------------------------------- timeout

def test_timeout_en_cours_de_generation(cfg):
    ferme = threading.Event()
    c, _ = client(cfg, texte(*["x"] * 200, delai=0.05, ferme=ferme))
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        c.chat(MESSAGES, timeout=0.3)
    assert time.monotonic() - t0 < 1.0
    assert ferme.wait(2.0), "le flux doit être fermé pour arrêter la génération côté serveur"


def test_timeout_avant_le_premier_morceau(cfg):
    c, _ = client(cfg, texte("trop tard", avant=3.0))
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        c.chat(MESSAGES, timeout=0.3)
    assert time.monotonic() - t0 < 1.0


def test_le_timeout_vient_de_la_config_par_defaut():
    cfg = ConfigNyx.creer(TIMEOUT_LLM=0.3)
    c, _ = client(cfg, texte("trop tard", avant=3.0))
    with pytest.raises(LLMTimeout):
        c.chat(MESSAGES)


def test_timeout_zero_signifie_illimite():
    cfg = ConfigNyx.creer(TIMEOUT_LLM=0.0)
    c, _ = client(cfg, texte("a", "b", "c", delai=0.1))
    assert c.chat(MESSAGES).texte == "abc"


# ---------------------------------------------------------------- annulation

def test_annulation_par_evenement(cfg):
    ferme = threading.Event()
    c, _ = client(cfg, texte(*["x"] * 200, delai=0.05, ferme=ferme))
    stop = threading.Event()
    threading.Timer(0.15, stop.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMAnnule):
        c.chat(MESSAGES, annulation=stop)
    assert time.monotonic() - t0 < 0.6
    assert ferme.wait(2.0), "l'annulation doit fermer le flux"


def test_annuler_tout_et_nettoyage_des_appels_en_cours(cfg):
    c, _ = client(cfg, texte(*["x"] * 200, delai=0.05))
    resultat = {}

    def appel():
        try:
            c.chat(MESSAGES)
        except Exception as e:  # noqa: BLE001
            resultat["erreur"] = e
    t = threading.Thread(target=appel)
    t.start()
    for _ in range(50):
        if c.appels_en_cours:
            break
        time.sleep(0.01)
    assert c.appels_en_cours == 1
    c.annuler_tout()
    t.join(2.0)
    assert isinstance(resultat.get("erreur"), LLMAnnule)
    assert c.appels_en_cours == 0


def test_apres_une_annulation_les_appels_suivants_fonctionnent(cfg):
    c, _ = client(cfg, [texte(*["x"] * 100, delai=0.05), texte("ok")])
    threading.Timer(0.1, c.annuler_tout).start()
    with pytest.raises(LLMAnnule):
        c.chat(MESSAGES)
    assert c.chat(MESSAGES).texte == "ok"


# ---------------------------------------------------------------- erreurs et nouvelles tentatives

def test_ollama_eteint_donne_llmindisponible_apres_les_tentatives(cfg):
    c, b = client(cfg, echec(ConnectionError("refusé")), tentatives=3)
    with pytest.raises(LLMIndisponible) as info:
        c.chat(MESSAGES)
    assert len(b.appels) == 3 and "ollama serve" in str(info.value)
    assert isinstance(info.value.__cause__, ConnectionError)


def test_erreur_de_connexion_pendant_le_flux(cfg):
    c, b = client(cfg, echec(ConnectionError("coupé"), pendant_le_flux=True), tentatives=2)
    with pytest.raises(LLMIndisponible):
        c.chat(MESSAGES)
    assert len(b.appels) == 2


def test_reussite_a_la_deuxieme_tentative(cfg):
    c, b = client(cfg, [echec(ConnectionError("démarrage")), texte("prêt")], tentatives=2)
    assert c.chat(MESSAGES).texte == "prêt" and len(b.appels) == 2


def test_modele_introuvable_sans_nouvelle_tentative(cfg):
    class ErreurReponse(Exception):
        status_code = 404
    c, b = client(cfg, echec(ErreurReponse("model not found")), tentatives=3)
    with pytest.raises(ModeleIntrouvable) as info:
        c.chat(MESSAGES, modele="inconnu:1b")
    assert "ollama pull inconnu:1b" in str(info.value) and len(b.appels) == 1


def test_vraie_erreur_ollama_404():
    ollama = pytest.importorskip("ollama")
    c, _ = client(ConfigNyx.defauts(), echec(ollama.ResponseError("model 'x' not found", 404)))
    with pytest.raises(ModeleIntrouvable):
        c.chat(MESSAGES)


def test_erreur_inconnue_devient_erreurllm(cfg):
    c, b = client(cfg, echec(ValueError("format inattendu")), tentatives=3)
    with pytest.raises(ErreurLLM) as info:
        c.chat(MESSAGES)
    assert "ValueError" in str(info.value) and len(b.appels) == 1
    assert not isinstance(info.value, (LLMIndisponible, LLMTimeout))


def test_annulation_pendant_la_pause_entre_deux_tentatives(cfg):
    backend = FauxOllama(echec(ConnectionError("x")))
    c = ClientLLM(cfg, backend=backend, tentatives=2, pause_tentative=5.0)
    stop = threading.Event()
    threading.Timer(0.15, stop.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMAnnule):
        c.chat(MESSAGES, annulation=stop)
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------- liste des modèles

def test_lister_modeles_ancienne_et_nouvelle_forme():
    cfg = ConfigNyx.defauts()
    ancien = ClientLLM(cfg, backend=FauxOllama(texte("x"), liste={"models": [{"name": "a:1"}, {"model": "b:2"}]}))
    assert ancien.lister_modeles() == ["a:1", "b:2"]

    class Mod:
        def __init__(self, model):
            self.model = model

    class Liste:
        models = [Mod("c:3"), Mod("d:4")]
    recent = ClientLLM(cfg, backend=FauxOllama(texte("x"), liste=Liste()))
    assert recent.lister_modeles() == ["c:3", "d:4"]


def test_lister_modeles_ollama_eteint_et_bloque():
    cfg = ConfigNyx.defauts()
    eteint = ClientLLM(cfg, backend=FauxOllama(texte("x"), liste=ConnectionError("refusé")))
    with pytest.raises(LLMIndisponible):
        eteint.lister_modeles()
    bloque = ClientLLM(cfg, backend=FauxOllama(texte("x"), liste=lambda: time.sleep(3)))
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        bloque.lister_modeles(timeout=0.2)
    assert time.monotonic() - t0 < 1.0


def test_modele_installe():
    installes = ["qwen2.5:7b", "nomic-embed-text:latest"]
    assert ClientLLM.modele_installe("qwen2.5:7b", installes)
    assert ClientLLM.modele_installe("nomic-embed-text", installes)       # :latest implicite
    assert not ClientLLM.modele_installe("qwen2.5:3b", installes)
    assert not ClientLLM.modele_installe("qwen2.5", installes)


def test_verifier():
    cfg = ConfigNyx.creer(model_name="qwen2.5:7b", model_consolidation="gemma2:2b")
    c = ClientLLM(cfg, backend=FauxOllama(texte("x")))
    v = c.verifier()
    assert v["joignable"] and v["manquants"] == ["gemma2:2b"] and "ollama pull gemma2:2b" in v["message"]
    assert c.verifier(["qwen2.5:7b"])["manquants"] == []
    eteint = ClientLLM(cfg, backend=FauxOllama(texte("x"), liste=ConnectionError("x")))
    v = eteint.verifier()
    assert v["joignable"] is False and v["manquants"] == ["qwen2.5:7b", "gemma2:2b"] and "ollama serve" in v["message"]


# ---------------------------------------------------------------- extraire_json

@pytest.mark.parametrize("brut, attendu", [
    ('{"valence": 0.5}', {"valence": 0.5}),
    ('```json\n{"valence": -0.2, "activation": 0.4}\n```', {"valence": -0.2, "activation": 0.4}),
    ('Voici mon analyse : {"concepts": ["a", "b"]} Voilà.', {"concepts": ["a", "b"]}),
    ('```\n{"x": 1}\n```', {"x": 1}),
    ('{"a": {"b": 2}}', {"a": {"b": 2}}),
])
def test_extraire_json_cas_valides(brut, attendu):
    assert extraire_json(brut) == attendu


@pytest.mark.parametrize("brut", ["", "   ", "pas de json", "{cassé", "[1, 2, 3]", "42", None, 123])
def test_extraire_json_ne_leve_jamais(brut):
    assert extraire_json(brut) is None


# ---------------------------------------------------------------- bibliothèque réelle

def test_ollama_absent_donne_un_message_clair(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ollama", None)  # `import ollama` lève ImportError
    with pytest.raises(ImportError, match="pip install ollama"):
        _importer_ollama()


def test_vraie_bibliotheque_ollama_injoignable():
    """Vérifie, avec la vraie bibliothèque, que l'erreur de connexion est bien reconnue."""
    ollama = pytest.importorskip("ollama")
    reel = ollama.Client(host="http://127.0.0.1:9")  # port fermé
    c = ClientLLM(ConfigNyx.defauts(), backend=reel, tentatives=2, pause_tentative=0.01)
    t0 = time.monotonic()
    with pytest.raises(LLMIndisponible):
        c.chat(MESSAGES)
    with pytest.raises(LLMIndisponible):
        c.lister_modeles()
    assert time.monotonic() - t0 < 5.0
    assert c.verifier()["joignable"] is False

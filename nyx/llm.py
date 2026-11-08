"""
nyx/llm.py : client du modèle de langage (module 2/9).

Remplace `_appeler_llm`, `_lister_modeles_ollama` et `_modele_installe` de l'ancien main.py.

Ce que l'ancien appel `ollama.chat(...)` ne savait pas faire :
  - s'arrêter : un modèle figé bloquait Nyx pour toujours  -> timeout total (config.TIMEOUT_LLM)
  - être annulé depuis l'interface                        -> `annulation` / `annuler_tout()`
  - montrer la réponse au fur et à mesure                 -> callback `sur_morceau`
  - dire clairement ce qui ne va pas                      -> erreurs typées
  - libérer la VRAM quand on ne l'utilise plus            -> keep_alive (partage avec la voix Qwen3-TTS)

La réponse garde la forme de l'ancien code : `reponse['message']['content']`, donc les appelants
existants n'ont rien à changer. Le modèle, `num_ctx` et le timeout sont relus dans la config à
chaque appel.

Nouveauté (Qwen3-TTS) :
  - `keep_alive` est envoyé à Ollama pour libérer la VRAM quand Nyx ne parle pas.
  - Valeur par défaut : cfg.ollama_keep_alive ("30s").
  - On peut le surcharger par appel via le paramètre `keep_alive`.
"""
import json
import logging
import queue
import re
import threading
import time

_log = logging.getLogger("nyx.llm")
_PAS = 0.05  # période de surveillance (annulation / échéance), en secondes


# ---------------------------------------------------------------- erreurs

class ErreurLLM(Exception):
    """Erreur générique liée au modèle."""


class LLMIndisponible(ErreurLLM):
    """Ollama est éteint ou injoignable."""


class ModeleIntrouvable(ErreurLLM):
    """Le modèle demandé n'est pas installé."""


class LLMTimeout(ErreurLLM):
    """L'appel a dépassé la durée autorisée."""


class LLMAnnule(ErreurLLM):
    """L'appel a été annulé par l'utilisateur."""


_NOMS_CONNEXION = {"ConnectError", "ConnectTimeout", "ReadError", "RemoteProtocolError",
                   "WriteError", "NetworkError"}


def _est_connexion(e):
    return isinstance(e, ConnectionError) or type(e).__name__ in _NOMS_CONNEXION


def _traduire(e, modele):
    """Transforme une exception de la bibliothèque en erreur Nyx lisible."""
    if isinstance(e, ErreurLLM):
        return e
    if getattr(e, "status_code", None) == 404:
        return ModeleIntrouvable(
            f"Modèle introuvable : {modele}. Installez-le avec :  ollama pull {modele}")
    if _est_connexion(e):
        return LLMIndisponible(
            "Ollama est injoignable. Est-il lancé ? (commande : ollama serve)")
    if "Timeout" in type(e).__name__:
        return LLMTimeout(f"La bibliothèque a signalé un délai dépassé ({type(e).__name__}).")
    return ErreurLLM(f"{type(e).__name__} : {e}")


# ---------------------------------------------------------------- réponse

class ReponseLLM(dict):
    """
    Réponse compatible avec l'ancien code (`rep['message']['content']`), avec des informations en plus.
        .texte               contenu de la réponse
        .modele              modèle utilisé
        .duree_s             durée totale de l'appel
        .premier_morceau_s   délai avant le premier morceau
        .morceaux            nombre de morceaux reçus
    """

    def __init__(self, texte, modele, duree_s, premier_morceau_s, morceaux):
        super().__init__(message={"role": "assistant", "content": texte}, model=modele)
        self.modele, self.duree_s = modele, duree_s
        self.premier_morceau_s, self.morceaux = premier_morceau_s, morceaux

    @property
    def texte(self):
        return self["message"]["content"]


def _contenu(morceau):
    """Texte d'un morceau, que ce soit un dict (ancienne lib) ou un objet ChatResponse (récente)."""
    try:
        contenu = morceau["message"]["content"]
    except (KeyError, TypeError, AttributeError):
        return ""
    return contenu if isinstance(contenu, str) else ""


# ---------------------------------------------------------------- utilitaires

def _importer_ollama():
    try:
        import ollama
    except ImportError as e:
        raise ImportError("La bibliothèque 'ollama' est requise :  pip install ollama") from e
    return ollama


def _appel_borne(fonction, delai, libelle):
    """Exécute `fonction` dans un thread et abandonne l'attente après `delai` secondes."""
    resultat = {}

    def cible():
        try:
            resultat["valeur"] = fonction()
        except BaseException as e:  # noqa: BLE001
            resultat["erreur"] = e

    t = threading.Thread(target=cible, daemon=True)
    t.start()
    t.join(delai)
    if t.is_alive():
        raise LLMTimeout(f"{libelle} : pas de réponse après {delai:.0f} s.")
    if "erreur" in resultat:
        raise resultat["erreur"]
    return resultat["valeur"]


_RE_BLOC_CODE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extraire_json(texte):
    """
    Extrait un objet JSON d'une sortie de modèle, même entourée de texte ou de ```json ... ```.
    Retourne un dict, ou None si rien d'exploitable. Ne lève jamais.
    """
    if not isinstance(texte, str) or not texte.strip():
        return None
    candidats = [texte.strip()]
    candidats += [m.strip() for m in _RE_BLOC_CODE.findall(texte)]
    debut, fin = texte.find("{"), texte.rfind("}")
    if 0 <= debut < fin:
        candidats.append(texte[debut:fin + 1])
    for candidat in candidats:
        try:
            valeur = json.loads(candidat)
        except ValueError:
            continue
        if isinstance(valeur, dict):
            return valeur
    return None


# ---------------------------------------------------------------- client

class ClientLLM:
    """
    Client Ollama. `cfg` est un ConfigNyx (relu à chaque appel).
    `backend` permet d'injecter un faux Ollama dans les tests.

    Nouveauté : `keep_alive` est envoyé à Ollama à chaque appel, pour libérer la VRAM
    après un délai d'inactivité. Utile pour cohabiter avec Qwen3-TTS sur GPU limité.
    """

    def __init__(self, cfg, backend=None, tentatives=2, pause_tentative=0.5):
        self.cfg = cfg
        self._backend = backend if backend is not None else _importer_ollama()
        self.tentatives = max(1, int(tentatives))
        self.pause_tentative = pause_tentative
        self._verrou = threading.Lock()
        self._actifs = set()   # un Event d'annulation par appel en cours
        self.dernier = {}      # statistiques du dernier appel réussi

    # ------------------------------------------------------------ annulation

    def annuler_tout(self):
        """Annule tous les appels en cours (bouton « Annuler » de l'interface)."""
        with self._verrou:
            for evenement in self._actifs:
                evenement.set()

    @property
    def appels_en_cours(self):
        with self._verrou:
            return len(self._actifs)

    # ------------------------------------------------------------ conversation

    def chat(self, messages, format=None, temperature=None, modele=None, timeout=None,
             annulation=None, sur_morceau=None, penser=None, max_tokens=None,
             keep_alive=None):
        """
        Envoie une conversation et retourne une ReponseLLM.

        timeout      secondes au total (None = config.TIMEOUT_LLM, 0 = illimité)
        annulation   threading.Event optionnel : le positionner annule l'appel
        sur_morceau  callback(delta: str) appelé dans le thread appelant
        penser       False désactive le mode « réflexion » des modèles qui en ont un
        max_tokens   plafond de longueur de la réponse
        keep_alive   durée de rétention du modèle en VRAM après l'appel
                     (None = cfg.ollama_keep_alive, "30s" par défaut)
        Lève LLMTimeout, LLMAnnule, LLMIndisponible, ModeleIntrouvable ou ErreurLLM.
        """
        nom_modele = modele or self.cfg.model_name
        options = {"num_ctx": self.cfg.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)

        # keep_alive : explicite > config > défaut Ollama
        if keep_alive is None:
            keep_alive = getattr(self.cfg, "ollama_keep_alive", "30s")

        parametres = {
            "model": nom_modele,
            "messages": messages,
            "options": options,
            "stream": True,
            "keep_alive": keep_alive,
        }
        if format:
            parametres["format"] = format
        if penser is not None:
            parametres["think"] = penser

        delai = self.cfg.TIMEOUT_LLM if timeout is None else timeout
        echeance = time.monotonic() + delai if delai else None
        interne = threading.Event()
        with self._verrou:
            self._actifs.add(interne)
        debut = time.monotonic()
        try:
            for tentative in range(1, self.tentatives + 1):
                try:
                    reponse = self._tentative(parametres, nom_modele, echeance, delai, interne,
                                              annulation, sur_morceau, debut)
                    self.dernier = {"modele": nom_modele, "duree_s": reponse.duree_s,
                                    "premier_morceau_s": reponse.premier_morceau_s,
                                    "caracteres": len(reponse.texte), "morceaux": reponse.morceaux}
                    return reponse
                except LLMIndisponible:
                    if tentative == self.tentatives:
                        raise
                    _log.warning("Ollama injoignable (tentative %d/%d), nouvel essai…",
                                 tentative, self.tentatives)
                    self._pause(self.pause_tentative, echeance, delai, interne, annulation)
        finally:
            with self._verrou:
                self._actifs.discard(interne)

    def chat_texte(self, messages, **kw):
        """Raccourci : retourne directement le texte de la réponse."""
        return self.chat(messages, **kw).texte

    # ------------------------------------------------------------ cœur

    @staticmethod
    def _interrompu(echeance, delai, interne, annulation):
        """Lève LLMAnnule / LLMTimeout si l'appel doit s'arrêter."""
        if interne.is_set() or (annulation is not None and annulation.is_set()):
            raise LLMAnnule("Appel au modèle annulé.")
        if echeance is not None and time.monotonic() >= echeance:
            raise LLMTimeout(f"Le modèle n'a pas terminé en {delai:.0f} s.")

    def _pause(self, secondes, echeance, delai, interne, annulation):
        fin = time.monotonic() + secondes
        while time.monotonic() < fin:
            self._interrompu(echeance, delai, interne, annulation)
            time.sleep(min(_PAS, max(0.0, fin - time.monotonic())))

    def _tentative(self, parametres, nom_modele, echeance, delai, interne, annulation,
                   sur_morceau, debut):
        file = queue.Queue()
        stop = threading.Event()

        def travail():
            flux = None
            try:
                flux = self._backend.chat(**parametres)
                for morceau in flux:
                    if stop.is_set():
                        break
                    file.put(("morceau", morceau))
                file.put(("fin", None))
            except BaseException as e:  # noqa: BLE001
                file.put(("erreur", e))
            finally:
                fermer = getattr(flux, "close", None)
                if callable(fermer):
                    try:
                        fermer()
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=travail, daemon=True, name="nyx-llm").start()
        parties, premier, callback = [], None, sur_morceau
        try:
            while True:
                self._interrompu(echeance, delai, interne, annulation)
                try:
                    genre, valeur = file.get(timeout=_PAS)
                except queue.Empty:
                    continue
                if genre == "erreur":
                    raise _traduire(valeur, nom_modele) from valeur
                if genre == "fin":
                    break
                texte = _contenu(valeur)
                if premier is None:
                    premier = time.monotonic() - debut
                if texte:
                    parties.append(texte)
                    if callback is not None:
                        try:
                            callback(texte)
                        except Exception:  # noqa: BLE001
                            _log.exception("sur_morceau a échoué : callback désactivé")
                            callback = None
        finally:
            stop.set()
        return ReponseLLM("".join(parties), nom_modele, time.monotonic() - debut,
                          premier if premier is not None else 0.0, len(parties))

    # ------------------------------------------------------------ modèles installés

    def lister_modeles(self, timeout=10.0):
        """Noms des modèles installés (compatible anciennes et nouvelles versions de la bibliothèque)."""
        try:
            rep = _appel_borne(self._backend.list, timeout, "Liste des modèles")
        except ErreurLLM:
            raise
        except Exception as e:  # noqa: BLE001
            raise _traduire(e, "") from e
        modeles = rep["models"] if isinstance(rep, dict) else getattr(rep, "models", [])
        noms = []
        for mod in modeles or []:
            if isinstance(mod, dict):
                nom = mod.get("name") or mod.get("model")
            else:
                nom = getattr(mod, "model", None) or getattr(mod, "name", None)
            if nom:
                noms.append(nom)
        return noms

    @staticmethod
    def modele_installe(nom, installes):
        return nom in installes or (":" not in nom and f"{nom}:latest" in installes)

    def verifier(self, requis=None):
        """
        Diagnostic rapide : Ollama répond-il, et les modèles nécessaires sont-ils installés ?
        Retourne {"joignable", "installes", "manquants", "message"}. Ne lève jamais.
        """
        requis = list(dict.fromkeys(requis or [self.cfg.model_name, self.cfg.model_consolidation]))
        try:
            installes = self.lister_modeles()
        except ErreurLLM as e:
            return {"joignable": False, "installes": [], "manquants": requis, "message": str(e)}
        manquants = [m for m in requis if not self.modele_installe(m, installes)]
        message = ("Ollama joignable, modèles prêts." if not manquants else
                   "Modèle(s) à installer : " + ", ".join(f"ollama pull {m}" for m in manquants))
        return {"joignable": True, "installes": installes, "manquants": manquants, "message": message}

    # ------------------------------------------------------------ gestion VRAM

    def decharger_modele(self, modele=None):
        """
        Demande à Ollama de décharger un modèle de la VRAM immédiatement.
        Utile avant de charger Qwen3-TTS si la VRAM est serrée.
        Ne lève jamais : log et retourne False si Ollama ne répond pas.
        """
        modele = modele or self.cfg.model_name
        try:
            # Ollama décharge un modèle quand on lui envoie keep_alive=0 sans prompt.
            self._backend.chat(
                model=modele,
                messages=[{"role": "user", "content": ""}],
                keep_alive=0,
                stream=False,
                options={"num_predict": 1, "num_ctx": 1},
            )
            _log.info("Modèle Ollama déchargé : %s", modele)
            return True
        except Exception as e:  # noqa: BLE001
            _log.debug("Déchargement Ollama impossible : %s", e)
            return False

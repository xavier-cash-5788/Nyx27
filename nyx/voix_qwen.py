"""
nyx/voix_qwen.py : synthèse vocale via Qwen3-TTS (0.6B CustomVoice) pour Nyx.

Qwen3-TTS est un modèle TTS local haute qualité de Qwen (Alibaba), qui tourne sur GPU.
Il offre 9 voix intégrées et supporte 10 langues dont le français.

Ce module expose :
  - QwenVoix    : wrapper direct autour de Qwen3-TTS
  - VoixHybride : combine QwenVoix (principal) + PiperVoix (fallback automatique)

API identique à PiperVoix (nyx/voix.py) pour interchangeabilité totale :
    dire(texte, bloquant=False)
    arreter()
    verifier() -> InfoVoix
    synthetiser(texte) -> (audio numpy, sample_rate)
    sauver_wav(texte, chemin)

Découpage intelligent :
  Les textes longs sont découpés en morceaux courts (< max_chars, 350 par défaut)
  et lus séquentiellement. Cela permet de lire 100 % d'un texte long (histoire,
  introspection) sans que Qwen3-TTS ne tronque à ~30 secondes d'audio.

Streaming (v2) :
  La lecture utilise un THREAD PRODUCTEUR (synthèse) + un CONSOMMATEUR (lecture).
  Pendant que le morceau N est lu, le morceau N+1 se synthétise en arrière-plan.
  Cela supprime les silences entre les morceaux : la voix est continue.

IMPORTANT — dtype :
  Qwen3-TTS DOIT être chargé en bfloat16 (ou float32). float16 provoque
  des valeurs inf/nan dans les tenseurs de probabilité → CUDA crash
  (« probability tensor contains either inf, nan or element < 0 »).

Dépendances :
    pip install qwen-tts soundfile
    # + PyTorch CUDA : voir guide d'installation
    # + sox : sudo apt install sox

Le modèle est chargé en eager (au démarrage) pour une première parole rapide.
En cas d'erreur (OOM, CUDA, etc.), VoixHybride bascule automatiquement sur Piper.
"""
import gc
import logging
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("nyx.voix_qwen")

# Pré-compilation des regex (performances)
_RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")
_RE_ESPACES = re.compile(r"\s+")
_RE_PHRASES = re.compile(r"(?<=[.!?…])\s+")


# ==================================================================== exceptions

class ErreurVoixQwen(Exception):
    """Erreur de synthèse ou de lecture audio avec Qwen3-TTS."""


class QwenIndisponible(ErreurVoixQwen):
    """Qwen3-TTS ou ses dépendances (PyTorch CUDA, qwen-tts) ne sont pas installés."""


class ModeleQwenIntrouvable(ErreurVoixQwen):
    """Le modèle Qwen3-TTS n'a pas pu être chargé."""


# ==================================================================== InfoVoix

@dataclass
class InfoVoix:
    """Résumé de l'état du moteur de voix (même structure que dans voix.py)."""
    disponible: bool
    modele: str
    chemin: str
    message: str


# ==================================================================== QwenVoix

class QwenVoix:
    """
    Wrapper autour de Qwen3-TTS pour lire du texte à voix haute.

    Utilisation :
        voix = QwenVoix(speaker="Sohee", language="French")
        voix.charger()                  # eager : charge le modèle (~5-30 s)
        info = voix.verifier()          # dit si tout est prêt
        voix.dire("Bonjour Xavier")     # synthétise et joue (streaming)
        voix.arreter()                  # arrête la lecture en cours

    Le modèle reste chargé en mémoire tant que l'instance existe.
    Les textes longs sont découpés en morceaux et lus en streaming.
    """

    def __init__(
        self,
        modele="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker="Sohee",
        language="French",
        device="cuda:0",
        dtype="bfloat16",
        max_chars=350,
    ):
        """
        modele     identifiant HuggingFace ou chemin local
        speaker    voix parmi : Vivian, Serena, Ryan, Aiden, Dylan, Eric, Ono_Anna, Sohee, Uncle_Fu
        language   langue parmi : Chinese, English, French, German, Italian, Portuguese,
                                 Spanish, Japanese, Korean, Russian
        device     cuda:0, cuda:1, cpu (cpu = très lent)
        dtype      bfloat16 (RECOMMANDÉ, obligatoire pour Qwen3-TTS), float32
                   Ne PAS utiliser float16 (provoque des crashs CUDA).
        max_chars  longueur max d'un MORCEAU (le texte est découpé en morceaux de cette taille)
        """
        self.modele_id = modele
        self.speaker = speaker
        self.language = language
        self.device = device
        # Garde-fou : on force bfloat16 si float16 est demandé
        if dtype == "float16":
            _log.warning("dtype='float16' non supporté par Qwen3-TTS → bascule sur 'bfloat16'.")
            dtype = "bfloat16"
        self.dtype = dtype
        self.max_chars = int(max_chars)

        self._model = None
        self._sample_rate = 24000  # valeur par défaut, mise à jour au chargement
        self._verrou = threading.RLock()
        self._stop = threading.Event()
        self._thread_lecture = None
        self._derniere_erreur = None

    # ---------------------------------------------------------------- chargement

    def charger(self, force=False):
        """
        Charge le modèle Qwen3-TTS en mémoire (eager).
        Bloque jusqu'à ce que le modèle soit prêt (5-30 s).
        Ne lève pas : stocke l'erreur dans self._derniere_erreur et retourne False.
        """
        with self._verrou:
            if self._model is not None and not force:
                return True
            try:
                import torch
                from qwen_tts import Qwen3TTSModel
            except ImportError as e:
                self._derniere_erreur = f"Dépendance manquante : {e}"
                _log.warning("Qwen3-TTS indisponible : %s", e)
                return False

            try:
                dtype_torch = {
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }.get(self.dtype, torch.bfloat16)

                _log.info("Chargement du modèle Qwen3-TTS : %s (device=%s, dtype=%s)",
                          self.modele_id, self.device, self.dtype)
                debut = time.monotonic()
                self._model = Qwen3TTSModel.from_pretrained(
                    self.modele_id,
                    device_map=self.device,
                    dtype=dtype_torch,
                )
                duree = time.monotonic() - debut
                _log.info("Qwen3-TTS chargé en %.1f s.", duree)
                self._derniere_erreur = None
                return True
            except Exception as e:  # noqa: BLE001
                self._model = None
                self._derniere_erreur = f"Chargement échoué : {e}"
                _log.warning("Impossible de charger Qwen3-TTS : %s", e)
                return False

    def decharger(self):
        """Libère le modèle et vide la VRAM (utile en cas de fallback)."""
        with self._verrou:
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()
            _log.info("Modèle Qwen3-TTS déchargé.")

    # ---------------------------------------------------------------- vérification

    def verifier(self):
        """Retourne un InfoVoix sans lever d'exception."""
        if self._model is None:
            raison = self._derniere_erreur or "Modèle non chargé (appelez charger())."
            return InfoVoix(
                disponible=False,
                modele=self.modele_id,
                chemin=self.device,
                message=f"Qwen3-TTS indisponible : {raison}",
            )
        return InfoVoix(
            disponible=True,
            modele=self.modele_id,
            chemin=f"{self.device} · voice={self.speaker} · {self.language}",
            message=f"Qwen3-TTS prêt : {self.speaker} ({self.language})",
        )

    # ---------------------------------------------------------------- découpage

    def decouper_en_phrases(self, texte, max_chars=None):
        """
        Découpe un texte long en morceaux courts (< max_chars) pour Qwen3-TTS.
        Chaque morceau se termine par une ponctuation forte si possible.
        Retourne une liste de chaînes.
        """
        max_chars = max_chars or self.max_chars
        if not texte:
            return []

        texte = _RE_EMOJI.sub("", texte)
        texte = _RE_ESPACES.sub(" ", texte).strip()
        if not texte:
            return []

        phrases = _RE_PHRASES.split(texte)

        morceaux = []
        courant = ""
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            # Si la phrase seule dépasse max_chars, on la coupe brutalement
            if len(phrase) > max_chars:
                if courant:
                    morceaux.append(courant)
                    courant = ""
                for i in range(0, len(phrase), max_chars):
                    morceaux.append(phrase[i:i + max_chars])
                continue
            # Si ajouter la phrase dépasse max_chars, on ferme le morceau
            if courant and len(courant) + 1 + len(phrase) > max_chars:
                morceaux.append(courant)
                courant = phrase
            else:
                courant = (courant + " " + phrase).strip() if courant else phrase

        if courant:
            morceaux.append(courant)

        return morceaux

    # ---------------------------------------------------------------- synthèse

    def synthetiser(self, texte):
        """
        Synthétise UN morceau de texte et retourne (audio numpy float32, sample_rate).
        Pour les textes longs, voir decouper_en_phrases().
        Lève ErreurVoixQwen en cas d'échec.
        """
        if not texte or not texte.strip():
            import numpy as np
            return np.zeros(0, dtype="float32"), self._sample_rate

        if self._model is None:
            raise ErreurVoixQwen(
                "Modèle Qwen3-TTS non chargé. Appelez charger() avant synthetiser()."
            )

        import numpy as np
        import torch

        texte = self._preparer_texte(texte)
        if not texte:
            return np.zeros(0, dtype="float32"), self._sample_rate

        try:
            wavs, sr = self._model.generate_custom_voice(
                text=texte,
                language=self.language,
                speaker=self.speaker,
            )
        except Exception as e:  # noqa: BLE001
            raise ErreurVoixQwen(f"Génération audio échouée : {e}") from e

        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype="float32")

        self._sample_rate = int(sr)
        return audio, self._sample_rate

    def _preparer_texte(self, texte):
        """
        Prépare un morceau de texte pour Qwen3-TTS : retire les emojis,
        normalise les espaces. Le découpage est fait en amont.
        """
        texte = _RE_EMOJI.sub("", texte)
        texte = _RE_ESPACES.sub(" ", texte).strip()
        return texte

    # ---------------------------------------------------------------- lecture

    def dire(self, texte, bloquant=False):
        """
        Synthétise et joue `texte`.
        - Non-bloquant par défaut (thread séparé).
        - Les textes longs sont découpés en morceaux et lus EN STREAMING :
          la synthèse du morceau suivant tourne pendant la lecture du morceau courant.
        """
        if not texte or not texte.strip():
            return
        morceaux = self.decouper_en_phrases(texte, max_chars=self.max_chars)
        if not morceaux:
            return
        if bloquant:
            self._lire_morceaux(morceaux)
        else:
            self._lancer_thread_morceaux(morceaux)

    def _lancer_thread_morceaux(self, morceaux):
        """Lance la lecture streaming dans un thread d'arrière-plan."""
        self.arreter()
        self._stop.clear()
        self._thread_lecture = threading.Thread(
            target=self._lire_morceaux,
            args=(morceaux,),
            daemon=True,
            name="nyx-voix-qwen",
        )
        self._thread_lecture.start()

    def _lire_morceaux(self, morceaux):
        """
        Lit chaque morceau avec STREAMING :
        un thread producteur synthétise les morceaux en arrière-plan,
        pendant que le thread courant (consommateur) les joue.
        Cela supprime les silences entre les morceaux.
        """
        import queue
        import sounddevice as sd

        file_synthese = queue.Queue()
        stop = self._stop

        def producteur():
            """Synthétise les morceaux un par un, en arrière-plan."""
            for i, morceau in enumerate(morceaux):
                if stop.is_set():
                    break
                try:
                    audio, sr = self.synthetiser(morceau)
                    if audio is not None and len(audio) > 0:
                        file_synthese.put(("audio", audio, sr))
                    else:
                        file_synthese.put(("vide", None, None))
                except ErreurVoixQwen as e:
                    _log.warning("Synthèse Qwen3-TTS impossible (morceau %d/%d) : %s",
                                 i + 1, len(morceaux), e)
                    file_synthese.put(("erreur", e, None))
                    return
            file_synthese.put(("fin", None, None))

        thread_prod = threading.Thread(target=producteur, daemon=True,
                                       name="nyx-voix-qwen-synthese")
        thread_prod.start()

        # Consommateur : joue les morceaux dès qu'ils sont prêts
        morceaux_lus = 0
        while True:
            if stop.is_set():
                break
            try:
                genre, audio, sr = file_synthese.get(timeout=0.1)
            except queue.Empty:
                continue

            if genre == "fin":
                break
            if genre == "erreur":
                _log.warning("Erreur de synthèse : %s", audio)
                break
            if genre == "vide":
                continue

            # Joue le morceau (pendant que le producteur synthétise le suivant)
            try:
                sd.play(audio, sr)
                while sd.get_stream().active:
                    if stop.is_set():
                        sd.stop()
                        return
                    time.sleep(0.05)
            except Exception as e:
                _log.warning("Erreur de lecture audio : %s", e)
                return
            morceaux_lus += 1

        if morceaux_lus == 0 and len(morceaux) > 0 and not stop.is_set():
            _log.warning("Aucun morceau n'a pu être lu.")

    def arreter(self):
        """Arrête la lecture en cours (si un thread joue)."""
        self._stop.set()
        if self._thread_lecture and self._thread_lecture.is_alive():
            self._thread_lecture.join(timeout=1.0)
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    def attendre(self, timeout=None):
        """Attend la fin de la lecture en cours."""
        if self._thread_lecture:
            self._thread_lecture.join(timeout=timeout)

    @property
    def en_lecture(self):
        return bool(self._thread_lecture and self._thread_lecture.is_alive())

    # ---------------------------------------------------------------- export

    def sauver_wav(self, texte, chemin):
        """
        Synthétise et écrit un fichier WAV.
        Découpe aussi le texte long en morceaux, puis concatène les audios.
        """
        import numpy as np
        morceaux = self.decouper_en_phrases(texte, max_chars=self.max_chars)
        if not morceaux:
            raise ErreurVoixQwen("Texte vide : rien à synthétiser.")
        tous_audios = []
        sr_final = self._sample_rate
        for morceau in morceaux:
            audio, sr = self.synthetiser(morceau)
            if audio is not None and len(audio) > 0:
                tous_audios.append(audio)
                sr_final = sr
        if not tous_audios:
            raise ErreurVoixQwen("Aucun audio généré.")
        audio = np.concatenate(tous_audios)
        data = np.clip(audio, -1.0, 1.0)
        data = (data * 32767).astype("int16")
        with wave.open(str(chemin), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr_final)
            w.writeframes(data.tobytes())
        return str(chemin)


# ==================================================================== VoixHybride

class VoixHybride:
    """
    Combine Qwen3-TTS (principal) et Piper (fallback).

    Comportement :
      - Essaie Qwen3-TTS d'abord.
      - Si Qwen3-TTS échoue (chargement, OOM, erreur à la synthèse),
        bascule automatiquement sur Piper pour la phrase en cours et les suivantes.
      - Retente périodiquement Qwen3-TTS (par défaut toutes les 5 minutes).

    Si `fallback` est None, aucune bascule n'est possible : on lève l'erreur.
    """

    def __init__(self, qwen, fallback=None, reessai_s=300):
        self.qwen = qwen
        self.fallback = fallback
        self.reessai_s = int(reessai_s)
        self._fallback_actif = False
        self._derniere_bascule = 0.0
        self._verrou = threading.RLock()

    @property
    def fallback_actif(self):
        return self._fallback_actif

    @property
    def en_lecture(self):
        if self._fallback_actif and self.fallback is not None:
            return self.fallback.en_lecture
        return self.qwen.en_lecture if self.qwen else False

    def charger(self):
        """Charge Qwen3-TTS. Si échec et fallback disponible, bascule sur Piper."""
        ok_qwen = self.qwen.charger() if self.qwen else False
        if ok_qwen:
            self._fallback_actif = False
            return True
        if self.fallback is not None:
            ok_piper = self.fallback.charger() if hasattr(self.fallback, "charger") else True
            if ok_piper:
                self._fallback_actif = True
                self._derniere_bascule = time.monotonic()
                _log.warning("Basculement initial sur Piper (Qwen3-TTS indisponible).")
                return True
        return False

    def verifier(self):
        """Retourne un InfoVoix (état combiné)."""
        info_qwen = self.qwen.verifier() if self.qwen else None
        info_piper = self.fallback.verifier() if self.fallback else None

        if self._fallback_actif and info_piper and info_piper.disponible:
            return InfoVoix(
                disponible=True,
                modele="Piper (fallback)",
                chemin=info_piper.chemin,
                message=f"Fallback Piper actif. Qwen3-TTS : "
                        f"{info_qwen.message if info_qwen else 'absent'}",
            )
        if info_qwen and info_qwen.disponible:
            return info_qwen
        if info_piper and info_piper.disponible:
            return InfoVoix(
                disponible=True,
                modele="Piper",
                chemin=info_piper.chemin,
                message="Qwen3-TTS indisponible. Piper prêt en secours.",
            )
        return InfoVoix(
            disponible=False,
            modele="Aucun",
            chemin="—",
            message="Ni Qwen3-TTS ni Piper ne sont disponibles.",
        )

    def dire(self, texte, bloquant=False):
        """Essaie Qwen3-TTS. Si échec, bascule sur Piper et relance la lecture."""
        if not texte or not texte.strip():
            return

        if self._fallback_actif and self.fallback is not None:
            if time.monotonic() - self._derniere_bascule > self.reessai_s:
                self._tenter_retour_qwen()

        if not self._fallback_actif and self.qwen is not None:
            try:
                self.qwen.dire(texte, bloquant=bloquant)
                return
            except Exception as e:  # noqa: BLE001
                _log.warning("Erreur Qwen3-TTS à la lecture : %s", e)
                self._basculer_sur_piper()

        if self.fallback is not None:
            try:
                self.fallback.dire(texte, bloquant=bloquant)
            except Exception as e:  # noqa: BLE001
                _log.warning("Erreur Piper à la lecture : %s", e)
        else:
            raise ErreurVoixQwen("Aucun moteur de voix disponible.")

    def _basculer_sur_piper(self):
        """Décharge Qwen, active Piper."""
        with self._verrou:
            if self._fallback_actif:
                return
            _log.warning("Bascule sur Piper.")
            try:
                self.qwen.decharger()
            except Exception as e:  # noqa: BLE001
                _log.debug("Déchargement Qwen échoué : %s", e)
            self._fallback_actif = True
            self._derniere_bascule = time.monotonic()

    def _tenter_retour_qwen(self):
        """Tente de recharger Qwen3-TTS après une période de fallback."""
        with self._verrou:
            _log.info("Tentative de retour à Qwen3-TTS…")
            try:
                if self.qwen.charger():
                    self._fallback_actif = False
                    _log.info("Retour à Qwen3-TTS réussi.")
                else:
                    self._derniere_bascule = time.monotonic()
            except Exception as e:  # noqa: BLE001
                _log.warning("Retour à Qwen3-TTS échoué : %s", e)
                self._derniere_bascule = time.monotonic()

    def arreter(self):
        """Arrête la lecture en cours (sur les deux moteurs)."""
        if self.qwen is not None:
            try:
                self.qwen.arreter()
            except Exception:  # noqa: BLE001
                pass
        if self.fallback is not None:
            try:
                self.fallback.arreter()
            except Exception:  # noqa: BLE001
                pass

    def synthetiser(self, texte):
        """Retourne (audio numpy, sample_rate) via le moteur actif."""
        if not self._fallback_actif and self.qwen is not None:
            try:
                return self.qwen.synthetiser(texte)
            except Exception as e:  # noqa: BLE001
                _log.warning("Synthèse Qwen échouée : %s", e)
                self._basculer_sur_piper()
        if self.fallback is not None:
            return self.fallback.synthetiser(texte)
        raise ErreurVoixQwen("Aucun moteur de voix disponible.")

    def sauver_wav(self, texte, chemin):
        """Synthétise dans un fichier WAV via le moteur actif."""
        if not self._fallback_actif and self.qwen is not None:
            try:
                return self.qwen.sauver_wav(texte, chemin)
            except Exception as e:  # noqa: BLE001
                _log.warning("Sauvegarde WAV Qwen échouée : %s", e)
                self._basculer_sur_piper()
        if self.fallback is not None:
            return self.fallback.sauver_wav(texte, chemin)
        raise ErreurVoixQwen("Aucun moteur de voix disponible.")


# ==================================================================== fabrique

def creer_voix_qwen(cfg=None, piper_fallback=None, **surcharges):
    """
    Fabrique utilisée par l'agent.

    Lit les paramètres depuis ConfigNyx si `cfg` est fourni.
    `piper_fallback` est une instance de PiperVoix déjà construite, ou None.
    """
    params = {
        "modele": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "speaker": "Sohee",
        "language": "French",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_chars": 350,
    }
    if cfg is not None:
        params["modele"] = getattr(cfg, "voix_qwen_modele", params["modele"])
        params["speaker"] = getattr(cfg, "voix_qwen_speaker", params["speaker"])
        params["language"] = getattr(cfg, "voix_qwen_language", params["language"])
        params["device"] = getattr(cfg, "voix_qwen_device", params["device"])
        params["dtype"] = getattr(cfg, "voix_qwen_dtype", params["dtype"])
        params["max_chars"] = getattr(cfg, "voix_qwen_max_chars", params["max_chars"])
    params.update(surcharges)

    qwen = QwenVoix(**params)

    if piper_fallback is None:
        return qwen

    reessai = getattr(cfg, "voix_qwen_reessai_s", 300) if cfg else 300
    return VoixHybride(qwen, fallback=piper_fallback, reessai_s=reessai)


# ==================================================================== test rapide

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Test de QwenVoix (Sohee, FR) avec streaming…")
    voix = QwenVoix(speaker="Sohee", language="French", max_chars=350, dtype="bfloat16")
    if not voix.charger():
        print(f"❌ Échec du chargement : {voix.verifier().message}")
        sys.exit(1)
    print(f"✅ {voix.verifier().message}")

    # Test court
    print("\n--- Test court ---")
    voix.dire("Bonjour Xavier, je suis Nyx. Ma nouvelle voix fonctionne.", bloquant=True)

    # Test long : un texte de plusieurs phrases (découpage + streaming)
    print("\n--- Test long (streaming) ---")
    texte_long = (
        "Il était une fois un nommé Roméo qui vivait dans une forêt inconnue. "
        "La forêt était pleine de mystères et de légendes mystérieuses. "
        "Un jour, alors que Roméo errait dans les bois, il croisa Nyx, "
        "une jeune femme avec des yeux d'un vert émeraude profond. "
        "Nyx possédait l'habitude de suivre la trace de l'évasion du roi Loup "
        "qui vivait à l'autre bout de la forêt. "
        "Ils décidèrent de travailler ensemble pour retrouver le prince mystérieux. "
        "Leur voyage les mena à travers des contrées oubliées, "
        "où chaque arbre murmurait des secrets anciens. "
        "Ensemble, ils découvrirent que le prince n'était pas perdu, "
        "mais qu'il avait choisi de disparaître pour protéger un secret bien plus grand. "
        "Et c'est ainsi que commença leur véritable aventure."
    )
    morceaux = voix.decouper_en_phrases(texte_long)
    print(f"Découpé en {len(morceaux)} morceaux.")
    print("Lecture streaming (pas de blanc entre les morceaux)…")
    t0 = time.monotonic()
    voix.dire(texte_long, bloquant=True)
    duree = time.monotonic() - t0
    print(f"Terminé en {duree:.1f} s.")

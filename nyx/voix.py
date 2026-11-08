"""
nyx/voix.py : synthèse vocale pour Nyx via Piper TTS.

Piper est un moteur TTS local, rapide, qui fonctionne sans GPU.
Ce module :
  - charge un modèle .onnx (par défaut fr_FR-siwis-medium) ;
  - synthétise un texte en audio et le joue ;
  - peut écrire dans un fichier WAV au lieu de jouer ;
  - supporte l'annulation (arrêter la lecture en cours) ;
  - s'exécute dans un thread pour ne pas bloquer l'interface.

Dépendances :
    pip install piper-tts sounddevice soundfile
Linux :
    sudo apt install libportaudio2 libsndfile1
"""
import logging
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("nyx.voix")


class ErreurVoix(Exception):
    """Erreur de synthèse ou de lecture audio."""


class VoixIndisponible(ErreurVoix):
    """Piper ou les dépendances audio ne sont pas installés."""


class ModeleVoixIntrouvable(ErreurVoix):
    """Le fichier .onnx du modèle n'existe pas."""


@dataclass
class InfoVoix:
    """Résumé de l'état du moteur de voix."""
    disponible: bool
    modele: str
    chemin: str
    message: str


class PiperVoix:
    """
    Wrapper autour de Piper pour lire du texte à voix haute.

    Utilisation :
        voix = PiperVoix(modele="models/fr_FR-siwis-medium.onnx")
        info = voix.verifier()          # dit si tout est prêt
        voix.dire("Bonjour Xavier")     # synthétise et joue
        voix.stop()                     # arrête la lecture en cours
    """

    def __init__(self, modele=None, dossier_modeles="models",
                 longueur_scale=None, bruit_scale=None, bruit_w=None,
                 volume=1.0, echantillonnage=22050):
        """
        modele             chemin complet vers le .onnx (sinon cherché dans dossier_modeles)
        dossier_modeles    dossier où chercher le modèle si `modele` est None
        longueur_scale     vitesse (>1 = plus lent, <1 = plus rapide)
        bruit_scale        variabilité de la voix (0 = monotone, 1 = expressif)
        bruit_w            variabilité de la durée des phonèmes
        volume             multiplicateur de volume (0.0 à 2.0)
        echantillonnage    fréquence d'échantillonnage (doit correspondre au modèle)
        """
        self.dossier_modeles = Path(dossier_modeles)
        self.modele = Path(modele) if modele else self._trouver_modele_defaut()
        self.longueur_scale = longueur_scale
        self.bruit_scale = bruit_scale
        self.bruit_w = bruit_w
        self.volume = max(0.0, float(volume))
        self.echantillonnage = int(echantillonnage)

        self._voice = None              # objet PiperVoice chargé (lazy)
        self._verrou = threading.RLock()
        self._stop = threading.Event()
        self._thread_lecture = None

    # ---------------------------------------------------------------- modèle

    def _trouver_modele_defaut(self):
        """Cherche un modèle .onnx dans le dossier configuré."""
        if not self.dossier_modeles.is_dir():
            return None
        candidats = sorted(self.dossier_modeles.glob("*.onnx"))
        # Priorité au modèle français si présent
        for c in candidats:
            if "fr_" in c.name.lower() or "french" in c.name.lower():
                return c
        return candidats[0] if candidats else None

    def verifier(self):
        """Retourne un InfoVoix sans lever d'exception."""
        if not self.modele or not Path(self.modele).is_file():
            return InfoVoix(
                disponible=False,
                modele=str(self.modele) if self.modele else "(aucun)",
                chemin=str(self.dossier_modeles),
                message=(f"Modèle introuvable : {self.modele}. "
                         f"Placez le .onnx et son .json dans {self.dossier_modeles}/."),
            )
        json_attendu = Path(str(self.modele) + ".json")
        if not json_attendu.is_file():
            return InfoVoix(
                disponible=False, modele=str(self.modele),
                chemin=str(self.dossier_modeles),
                message=f"Fichier de configuration manquant : {json_attendu.name}.",
            )
        try:
            import piper  # noqa: F401
        except ImportError:
            return InfoVoix(
                disponible=False, modele=str(self.modele),
                chemin=str(self.dossier_modeles),
                message="Piper n'est pas installé. Lancez : pip install piper-tts",
            )
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            return InfoVoix(
                disponible=False, modele=str(self.modele),
                chemin=str(self.dossier_modeles),
                message=("sounddevice n'est pas installé. "
                         "Lancez : pip install sounddevice soundfile"),
            )
        return InfoVoix(
            disponible=True, modele=str(self.modele),
            chemin=str(self.dossier_modeles),
            message=f"Voix prête : {Path(self.modele).name}",
        )

    def _charger(self):
        """Charge le modèle Piper en mémoire (lazy, thread-safe)."""
        if self._voice is not None:
            return self._voice
        with self._verrou:
            if self._voice is not None:
                return self._voice
            try:
                from piper import PiperVoice
            except ImportError as e:
                raise VoixIndisponible(
                    "Piper n'est pas installé : pip install piper-tts") from e
            if not self.modele or not Path(self.modele).is_file():
                raise ModeleVoixIntrouvable(f"Modèle introuvable : {self.modele}")
            _log.info("Chargement du modèle Piper : %s", self.modele)
            self._voice = PiperVoice.load(str(self.modele))
            _log.info("Modèle Piper chargé.")
            return self._voice

    # ---------------------------------------------------------------- synthèse

    def synthetiser(self, texte):
        """Retourne (numpy array float32, sample rate) sans jouer."""
        if not texte or not texte.strip():
            return None, self.echantillonnage
        try:
            import numpy as np
        except ImportError as e:
            raise VoixIndisponible("numpy requis (installé avec soundfile)") from e

        voice = self._charger()

        # Piper >= 1.2 : synthétise dans un flux audio
        try:
            from piper import SynthesisConfig
            config = SynthesisConfig(
                length_scale=self.longueur_scale,
                noise_scale=self.bruit_scale,
                noise_w_scale=self.bruit_w,
            )
            chunks = list(voice.synthesize(texte, syn_config=config))
        except (ImportError, TypeError):
            # Fallback anciennes versions (pas de SynthesisConfig)
            chunks = list(voice.synthesize(texte))

        if not chunks:
            return np.zeros(0, dtype="float32"), self.echantillonnage

        # Chaque chunk est un AudioChunk (audio_int16_bytes ou audio_float_array)
        audio_parts = []
        for chunk in chunks:
            if hasattr(chunk, "audio_float_array"):
                audio_parts.append(np.asarray(chunk.audio_float_array, dtype="float32"))
            elif hasattr(chunk, "audio_int16_bytes"):
                arr = np.frombuffer(chunk.audio_int16_bytes, dtype="int16").astype("float32") / 32768.0
                audio_parts.append(arr)
            elif isinstance(chunk, (bytes, bytearray)):
                arr = np.frombuffer(chunk, dtype="int16").astype("float32") / 32768.0
                audio_parts.append(arr)
        if not audio_parts:
            return np.zeros(0, dtype="float32"), self.echantillonnage
        audio = np.concatenate(audio_parts)
        if self.volume != 1.0:
            audio = np.clip(audio * self.volume, -1.0, 1.0)
        return audio, self.echantillonnage

    # ---------------------------------------------------------------- lecture

    def dire(self, texte, bloquant=False):
        """
        Synthétise et joue `texte`.
        - bloquant=True : attend la fin de la lecture.
        - bloquant=False : joue dans un thread d'arrière-plan.
        """
        if not texte or not texte.strip():
            return
        if bloquant:
            self._lire_bloquant(texte)
        else:
            self._lancer_thread(texte)

    def _lancer_thread(self, texte):
        self.stop()  # arrête la lecture précédente
        self._stop.clear()
        self._thread_lecture = threading.Thread(
            target=self._lire_bloquant, args=(texte,), daemon=True, name="nyx-voix")
        self._thread_lecture.start()

    def _lire_bloquant(self, texte):
        try:
            audio, sr = self.synthetiser(texte)
        except ErreurVoix as e:
            _log.warning("Synthèse vocale impossible : %s", e)
            return
        if audio is None or len(audio) == 0:
            return
        try:
            import sounddevice as sd
            sd.play(audio, sr)
            # Attend la fin, mais vérifie périodiquement si on doit s'arrêter
            while sd.get_stream().active:
                if self._stop.is_set():
                    sd.stop()
                    return
                time.sleep(0.05)
        except ImportError:
            _log.warning("sounddevice indisponible : lecture impossible.")
        except Exception as e:
            _log.warning("Erreur de lecture audio : %s", e)

    def stop(self):
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
        """Synthétise et écrit un fichier WAV."""
        import numpy as np
        audio, sr = self.synthetiser(texte)
        if audio is None or len(audio) == 0:
            raise ErreurVoix("Texte vide : rien à synthétiser.")
        data = np.clip(audio, -1.0, 1.0)
        data = (data * 32767).astype("int16")
        with wave.open(str(chemin), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(data.tobytes())
        return str(chemin)


def creer_voix(cfg=None, **surcharges):
    """
    Fabrique utilisée par l'agent.
    Lit les paramètres depuis ConfigNyx si `cfg` est fourni.
    """
    params = {
        "dossier_modeles": "models",
        "longueur_scale": None,
        "bruit_scale": None,
        "bruit_w": None,
        "volume": 1.0,
    }
    if cfg is not None:
        params["modele"] = getattr(cfg, "voix_modele", None) or None
        params["dossier_modeles"] = getattr(cfg, "voix_dossier", "models")
        params["longueur_scale"] = getattr(cfg, "voix_longueur", None)
        params["bruit_scale"] = getattr(cfg, "voix_bruit", None)
        params["volume"] = getattr(cfg, "voix_volume", 1.0)
    params.update(surcharges)
    return PiperVoix(**{k: v for k, v in params.items() if v is not None or k == "volume"})


# ---------------------------------------------------------------- test rapide

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    voix = PiperVoix(modele="models/fr_FR-siwis-medium.onnx")
    info = voix.verifier()
    print(info.message)
    if not info.disponible:
        sys.exit(1)
    print("Synthèse de test...")
    voix.dire("Bonjour, je suis Nyx. Ma voix fonctionne.", bloquant=True)
    print("Terminé.")
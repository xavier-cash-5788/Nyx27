"""
nyx/tts.py : la voix de Nyx via Qwen3-TTS (module optionnel).
Supporte les modèles 0.6B (léger) ou 1.7B (haute fidélité) en local.

Architecture miroir de llm.py :
  - erreurs typées (TTSIndisponible, VoixIntrouvable, ErreurTTS)
  - backend injectable pour les tests (par défaut : pipeline transformers)
  - streaming audio via callback `sur_morceau`
  - verrou thread (un seul appel TTS à la fois)
  - tolérant aux erreurs : ne lève jamais en production, log un warning

Le contrôle émotionnel est injecté via des instructions de style en anglais
(Qwen3-TTS les comprend mieux) : chaque émotion du système limbique (11 émotions)
est mappée vers un style vocal. L'intensité module la force de l'instruction.

La voix est produite en arrière-plan par l'agent (thread daemon) pour ne pas
bloquer l'affichage du texte. Les morceaux audio sont relayés à l'interface
via le callback `sur_evenement("audio_chunk", bytes)`.
"""
import io
import logging
import threading
import time
from dataclasses import dataclass

_log = logging.getLogger("nyx.tts")

# ------------------------------------------------------------------ mapping émotion → style vocal
# Les 11 émotions de nyx/limbique.py → instructions de style pour Qwen3-TTS.
# En anglais pour une meilleure compréhension par le modèle.
STYLE_EMOTION = {
    "Excitation joyeuse": "joyful, energetic, enthusiastic, fast-paced, bright",
    "Surprise agréable": "surprised, bright, curious, lively, wide-eyed",
    "Sérénité confiante": "calm, confident, serene, steady, assured",
    "Contentement calme": "peaceful, warm, gentle, soft, content",
    "Curiosité intense": "curious, inquisitive, engaged, attentive, eager",
    "Intérêt modéré": "neutral, thoughtful, reflective, moderate pace",
    "Colère": "angry, firm, assertive, intense, sharp",
    "Peur": "fearful, hesitant, cautious, trembling, soft",
    "Frustration": "frustrated, impatient, tense, sharp, clipped",
    "Tristesse": "sad, melancholic, soft, slow-paced, tender",
    "Neutralité": "neutral, factual, calm, clear, even-paced",
}

# ------------------------------------------------------------------ erreurs
class ErreurTTS(Exception):
    """Erreur générique liée à la synthèse vocale."""

class TTSIndisponible(ErreurTTS):
    """Le backend TTS n'a pas pu être chargé (dépendances manquantes, modèle absent...)."""

class VoixIntrouvable(ErreurTTS):
    """Le speaker demandé n'existe pas dans le modèle."""

# ------------------------------------------------------------------ réponse
@dataclass(frozen=True)
class ReponseTTS:
    """Résultat d'une synthèse vocale."""
    audio_bytes: bytes          # WAV complet (ou dernier chunk si streaming)
    duree_s: float              # durée totale de la synthèse
    modele: str                 # modèle utilisé
    voix: str                   # speaker utilisé
    emotion: str                # émotion appliquée
    sampling_rate: int = 24000  # taux d'échantillonnage (Qwen3-TTS = 24kHz)

# ------------------------------------------------------------------ backend par défaut
def _importer_transformers():
    """Importe `pipeline` de transformers. Lève ImportError si absent."""
    try:
        from transformers import pipeline
        return pipeline
    except ImportError as e:
        raise ImportError(
            "La bibliothèque 'transformers' est requise pour la voix : "
            "pip install transformers torch soundfile"
        ) from e


def _numpy_to_wav(audio_array, sampling_rate):
    """Convertit un tableau numpy en bytes WAV (sans écrire sur disque)."""
    try:
        import soundfile as sf
    except ImportError as e:
        raise TTSIndisponible(
            "La bibliothèque 'soundfile' est requise : pip install soundfile"
        ) from e
    tampon = io.BytesIO()
    sf.write(tampon, audio_array, sampling_rate, format="WAV")
    return tampon.getvalue()


class _BackendTransformers:
    """Backend par défaut : pipeline transformers (Qwen3-TTS)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._pipeline = None

    def charger(self):
        if self._pipeline is not None:
            return self._pipeline
        _log.info("Chargement du modèle TTS %s (peut prendre 1-2 min au premier lancement)...",
                  self.cfg.model_tts)
        try:
            pipeline = _importer_transformers()
            device = 0 if self.cfg.tts_use_gpu else -1
            self._pipeline = pipeline(
                "text-to-speech",
                model=self.cfg.model_tts,
                torch_dtype="float16",
                device=device,
            )
            _log.info("✅ Modèle TTS chargé sur %s", "GPU" if self.cfg.tts_use_gpu else "CPU")
            return self._pipeline
        except Exception as e:
            raise TTSIndisponible(f"Impossible de charger {self.cfg.model_tts} : {e}") from e

    def synthetise(self, prompt, speaker=None):
        """Appelle le pipeline et retourne (audio_array, sampling_rate)."""
        pipe = self.charger()
        params = {}
        if speaker:
            # Qwen3-TTS CustomVoice accepte un speaker prédéfini
            params["speaker"] = speaker
        try:
            sortie = pipe(prompt, **params)
        except Exception as e:
            err = str(e).lower()
            if "speaker" in err or "voice" in err:
                raise VoixIntrouvable(f"Speaker introuvable : {speaker}") from e
            raise TTSIndisponible(f"Échec de synthèse : {e}") from e
        audio = sortie.get("audio") if isinstance(sortie, dict) else sortie["audio"]
        sr = sortie.get("sampling_rate", 24000) if isinstance(sortie, dict) else 24000
        return audio, sr

# ------------------------------------------------------------------ client
class ClientTTS:
    """
    Client Qwen3-TTS. `cfg` est un ConfigNyx (relu à chaque appel).
    `backend` permet d'injecter un faux backend dans les tests.
    """

    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self._verrou = threading.Lock()
        self._backend = backend if backend is not None else _BackendTransformers(cfg)
        self.dernier = {}  # statistiques du dernier appel réussi

    # ------------------------------------------------------------ synthèse
    def parler(self, texte, emotion="Neutralité", intensite=None, sur_morceau=None):
        """
        Synthétise `texte` avec l'émotion du système limbique.
        emotion    : nom d'émotion (doit être dans STYLE_EMOTION, sinon "Neutralité")
        intensite  : 0.0 (plate) .. 1.0 (exagérée) ; None = cfg.intensite_emotion
        sur_morceau: callback(bytes) appelé à chaque chunk audio (streaming)
        Retourne une ReponseTTS.
        """
        if not texte or not texte.strip():
            return ReponseTTS(b"", 0.0, self.cfg.model_tts, self.cfg.voix_tts_speaker, emotion)

        intensite = self.cfg.intensite_emotion if intensite is None else intensite
        intensite = max(0.0, min(1.0, float(intensite)))
        prompt = self._construire_prompt(texte, emotion, intensite)

        debut = time.monotonic()
        try:
            with self._verrou:
                audio_array, sr = self._backend.synthetise(
                    prompt, speaker=self.cfg.voix_tts_speaker or None
                )
                audio_bytes = _numpy_to_wav(audio_array, sr)
            duree = time.monotonic() - debut

            # Streaming : on envoie le WAV complet en un seul morceau
            # (Qwen3-TTS ne supporte pas nativement le streaming morceau par morceau)
            if sur_morceau is not None:
                try:
                    sur_morceau(audio_bytes)
                except Exception:  # noqa: BLE001 - un bug d'affichage ne tue pas la synthèse
                    _log.exception("sur_morceau a échoué : callback désactivé")

            self.dernier = {
                "modele": self.cfg.model_tts,
                "voix": self.cfg.voix_tts_speaker,
                "emotion": emotion,
                "duree_s": duree,
                "taille_bytes": len(audio_bytes),
                "sampling_rate": sr,
            }
            return ReponseTTS(
                audio_bytes=audio_bytes,
                duree_s=duree,
                modele=self.cfg.model_tts,
                voix=self.cfg.voix_tts_speaker or "default",
                emotion=emotion,
                sampling_rate=sr,
            )
        except ErreurTTS:
            raise
        except Exception as e:
            raise TTSIndisponible(f"Erreur inattendue TTS : {e}") from e

    # ------------------------------------------------------------ prompt
    @staticmethod
    def _construire_prompt(texte, emotion, intensite):
        """Construit le prompt annoté pour Qwen3-TTS."""
        style = STYLE_EMOTION.get(emotion, STYLE_EMOTION["Neutralité"])
        # L'intensité module la longueur de l'instruction :
        #   0.0 → style minimal (juste "neutral")
        #   0.5 → style complet
        #   1.0 → style complet + adjectifs renforcés
        if intensite < 0.3:
            style_complet = "neutral, calm"
        elif intensite < 0.7:
            # Garde les 3 premiers adjectifs
            style_complet = ", ".join(style.split(", ")[:3])
        else:
            style_complet = style
        # Format d'instruction pour Qwen3-TTS
        return f"[{style_complet}] {texte}"

    # ------------------------------------------------------------ diagnostic
    def verifier(self):
        """
        Diagnostic rapide : le backend est-il chargeable ?
        Retourne {"disponible", "modele", "message"}. Ne lève jamais.
        """
        try:
            self._backend.charger()
            return {
                "disponible": True,
                "modele": self.cfg.model_tts,
                "message": f"Modèle TTS prêt : {self.cfg.model_tts}",
            }
        except TTSIndisponible as e:
            return {
                "disponible": False,
                "modele": self.cfg.model_tts,
                "message": str(e),
            }
        except Exception as e:
            return {
                "disponible": False,
                "modele": self.cfg.model_tts,
                "message": f"Erreur inattendue : {e}",
            }

# ------------------------------------------------------------------ faux backend pour les tests
class FauxBackendTTS:
    """
    Backend de test : produit un WAV silencieux de durée fixe.
    Enregistre les appels pour vérification.
    """

    def __init__(self, duree_s=0.5, sampling_rate=24000, echec=False):
        self.duree_s = duree_s
        self.sampling_rate = sampling_rate
        self.echec = echec
        self.appels = []

    def charger(self):
        if self.echec:
            raise TTSIndisponible("Backend TTS simulé en panne")
        return self

    def synthetise(self, prompt, speaker=None):
        self.appels.append({"prompt": prompt, "speaker": speaker})
        if self.echec:
            raise TTSIndisponible("Backend TTS simulé en panne")
        # WAV silencieux : tableau numpy de zéros
        try:
            import numpy as np
        except ImportError:
            # Repli sans numpy : tableau Python (soundfile l'accepte)
            nb_samples = int(self.duree_s * self.sampling_rate)
            audio = [0.0] * nb_samples
            return audio, self.sampling_rate
        nb_samples = int(self.duree_s * self.sampling_rate)
        audio = np.zeros(nb_samples, dtype=np.float32)
        return audio, self.sampling_rate

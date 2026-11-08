"""
nyx/limbique.py : le système limbique de Nyx (module 4/9).

Remplace, dans l'ancien main.py : _analyser_stimulus_emotionnel, _calculer_emotion_dominante,
_mettre_a_jour_emotions, _deriver_emotions, _generer_instruction_emotionnelle_concise, l'apaisement
émotionnel du sommeil et les effets du feedback sur l'humeur.

Modèle VAD : valence (-1 déplaisant .. +1 plaisant), activation (0 calme .. 1 excité),
dominance (0 soumis .. 1 maîtrise). Mêmes règles qu'avant (le test de parité le prouve), plus :
  - une réponse du modèle illisible ou non finie (NaN) ne produit plus une émotion extrême : repli neutre ;
  - un JSON entouré de texte ou de ```json ... ``` est lu correctement, même clés dans le désordre ;
  - l'analyse a son propre délai court (DELAI_ANALYSE) : un modèle lent ne bloque pas la conversation
    pour une simple étiquette émotionnelle ;
  - une annulation demandée par l'utilisateur (LLMAnnule) n'est PAS avalée par le repli neutre ;
  - le module n'affiche rien : il retourne des résultats que l'agent présente comme il veut.

L'état émotionnel vit ici ; la mémoire (module 3) reçoit l'humeur en paramètre via `humeur()`.
"""
import logging
import math
import re
import threading
from dataclasses import dataclass

from .llm import LLMAnnule, extraire_json

_log = logging.getLogger("nyx.limbique")

NEUTRE = (0.0, 0.5, 0.5)

# nom -> (emoji, ton, consigne). Source unique : l'interface et le prompt s'y réfèrent.
EMOTIONS = {
    "Excitation joyeuse": ("🎉", "enthousiaste", "Réponds avec énergie."),
    "Surprise agréable": ("😲", "étonné", "Exprime ta curiosité."),
    "Sérénité confiante": ("😊", "serein", "Réponds avec assurance."),
    "Contentement calme": ("😌", "paisible", "Sois bienveillant."),
    "Curiosité intense": ("🧐", "curieux", "Pose des questions."),
    "Intérêt modéré": ("🤔", "attentif", "Sois réfléchi."),
    "Colère": ("😠", "ferme", "Sois direct."),
    "Peur": ("😨", "prudent", "Sois hésitant."),
    "Frustration": ("😤", "impatient", "Va à l'essentiel."),
    "Tristesse": ("😢", "mélancolique", "Parle avec douceur."),
    "Neutralité": ("😐", "neutre", "Sois factuel."),
}
EMOJIS = {nom: emoji for nom, (emoji, _, _) in EMOTIONS.items()}


@dataclass(frozen=True)
class Humeur:
    """Instantané de l'état émotionnel."""
    valence: float
    activation: float
    dominance: float

    @property
    def vad(self):
        return (self.valence, self.activation, self.dominance)


@dataclass(frozen=True)
class AnalyseEmotion:
    """Résultat de l'analyse d'un message. `ok` est False quand le repli neutre a été utilisé."""
    valence: float
    activation: float
    dominance: float
    ok: bool = True
    source: str = "json"        # 'json' | 'nombres' | 'defaut'
    erreur: str = None

    @property
    def vad(self):
        return (self.valence, self.activation, self.dominance)


@dataclass(frozen=True)
class Perception:
    """Ce qui s'est passé lors du « filtre émotionnel » d'un tour de parole."""
    analyse: AnalyseEmotion
    avant: Humeur
    apres: Humeur
    emotion: str
    emoji: str


# ---------------------------------------------------------------- fonctions pures

def classer_emotion(valence, activation, dominance):
    """Émotion dominante (nom, emoji) pour un point VAD. Mêmes seuils que l'ancien code."""
    v, a, d = valence, activation, dominance
    if v > 0.3 and a > 0.6:
        nom = "Excitation joyeuse" if d > 0.6 else "Surprise agréable"
    elif v > 0.3 and a <= 0.6:
        nom = "Sérénité confiante" if d > 0.6 else "Contentement calme"
    elif -0.3 < v <= 0.3 and a > 0.6:
        nom = "Curiosité intense" if d > 0.6 else "Intérêt modéré"
    elif v <= -0.3 and a > 0.6:
        nom = "Colère" if d > 0.6 else "Peur"
    elif v <= -0.3 and a <= 0.6:
        nom = "Frustration" if d > 0.6 else "Tristesse"
    else:
        nom = "Neutralité"
    return nom, EMOJIS[nom]


def instruction_ton(nom, emoji=None):
    """Consigne de ton ajoutée au prompt système. Chaîne vide si l'émotion est inconnue."""
    if nom not in EMOTIONS:
        return ""
    emoji_defaut, ton, consigne = EMOTIONS[nom]
    return f"Ton actuel : {ton} {emoji or emoji_defaut}. {consigne}"


def construire_prompt_analyse(stimulus):
    return f"""Analyse le ton émotionnel du message suivant.
Repères d'échelle :
- "Bonjour" -> {{"valence": 0.1, "activation": 0.3, "dominance": 0.5}}
- "Super, j'ai réussi mon examen !" -> {{"valence": 0.7, "activation": 0.7, "dominance": 0.6}}
- "Tu es nul, je te déteste." -> {{"valence": -0.7, "activation": 0.7, "dominance": 0.6}}
Utilise toute l'échelle, mais réserve les valeurs extrêmes (±1) aux cas exceptionnels.
Message : "{stimulus}"
Réponds en JSON avec exactement ces clés : "valence" (-1 à 1), "activation" (0 à 1), "dominance" (0 à 1)."""


def _borner(v, a, d):
    return max(-1.0, min(1.0, v)), max(0.0, min(1.0, a)), max(0.0, min(1.0, d))


def lire_vad(contenu):
    """
    Extrait (valence, activation, dominance, source) d'une sortie de modèle, bornés dans leurs intervalles.
    Lève ValueError si la réponse est inexploitable ou contient des valeurs non finies.
    """
    contenu = contenu if isinstance(contenu, str) else ""
    data = extraire_json(contenu)
    if data is not None:
        try:
            v, a, d = float(data["valence"]), float(data["activation"]), float(data["dominance"])
            source = "json"
        except (KeyError, TypeError, ValueError):
            data = None
    if data is None:
        nombres = re.findall(r"-?\d+(?:\.\d+)?", contenu)   # repli : les 3 premiers nombres du texte
        if len(nombres) < 3:
            raise ValueError("réponse émotionnelle inexploitable")
        v, a, d = (float(n) for n in nombres[:3])
        source = "nombres"
    if not all(math.isfinite(x) for x in (v, a, d)):
        raise ValueError("valeurs émotionnelles non finies")
    return (*_borner(v, a, d), source)


# ---------------------------------------------------------------- système limbique

class Limbique:
    LISSAGE = 0.3               # poids d'un nouveau message dans l'humeur
    AMORTISSEMENT = 0.95        # à chaque message, l'humeur se rapproche un peu du neutre
    FEEDBACK_VALENCE = 0.2      # effet d'un 👍 / 👎 sur la valence
    FEEDBACK_ACTIVATION = 0.1   # effet d'un 👍 / 👎 sur l'activation
    DELAI_ANALYSE = 30.0        # secondes max pour l'analyse d'un message (0 = délai de la config)

    def __init__(self, cfg, llm=None, etat=NEUTRE):
        """
        cfg   ConfigNyx (TAU_EMOTION est relu à chaque dérive)
        llm   ClientLLM (module 2), ou tout objet avec .chat(messages, format=, temperature=, timeout=)
        etat  humeur initiale (valence, activation, dominance)
        """
        self.cfg = cfg
        self.llm = llm
        self._verrou = threading.RLock()
        self.valence, self.activation, self.dominance = etat

    # ------------------------------------------------------------ lecture de l'état

    def etat(self):
        with self._verrou:
            return Humeur(self.valence, self.activation, self.dominance)

    def humeur(self):
        """(valence, activation) : ce que la mémoire attend pour le rappel congruent à l'humeur."""
        with self._verrou:
            return (self.valence, self.activation)

    def reinitialiser(self):
        with self._verrou:
            self.valence, self.activation, self.dominance = NEUTRE

    def emotion_dominante(self, etat=None):
        etat = etat or self.etat()
        return classer_emotion(*etat.vad)

    def instruction_ton(self, etat=None):
        nom, emoji = self.emotion_dominante(etat)
        return instruction_ton(nom, emoji)

    # ------------------------------------------------------------ analyse d'un message

    def analyser(self, stimulus):
        """
        Demande au modèle le ton émotionnel d'un message. Ne lève jamais, sauf si l'utilisateur annule
        (LLMAnnule) : en cas d'échec (modèle indisponible, délai, réponse illisible), repli sur le neutre.
        """
        if self.llm is None:
            raise RuntimeError("Limbique sans client LLM : analyse impossible")
        try:
            reponse = self.llm.chat([{"role": "user", "content": construire_prompt_analyse(stimulus)}],
                                    format="json", temperature=0,
                                    timeout=self.DELAI_ANALYSE or None)
            v, a, d, source = lire_vad(reponse["message"]["content"])
            return AnalyseEmotion(v, a, d, True, source)
        except LLMAnnule:
            raise
        except Exception as e:  # noqa: BLE001 - tout échec d'analyse = repli neutre, comme avant
            _log.warning("analyse émotionnelle impossible : %s", e)
            return AnalyseEmotion(*NEUTRE, ok=False, source="defaut", erreur=str(e))

    def percevoir(self, stimulus):
        """Étape « filtre émotionnel » d'un tour : analyse le message puis met l'humeur à jour."""
        analyse = self.analyser(stimulus)
        avant = self.etat()
        self.mettre_a_jour(*analyse.vad)
        apres = self.etat()
        nom, emoji = self.emotion_dominante(apres)
        return Perception(analyse, avant, apres, nom, emoji)

    # ------------------------------------------------------------ dynamique de l'humeur

    def mettre_a_jour(self, v_stim, a_stim, d_stim):
        """Un message influence l'humeur (lissage), puis elle se rapproche légèrement du neutre."""
        f, k = self.LISSAGE, self.AMORTISSEMENT
        with self._verrou:
            self.valence = ((1 - f) * self.valence + f * v_stim) * k
            self.activation = ((1 - f) * self.activation + f * a_stim) * k + NEUTRE[1] * (1 - k)
            self.dominance = ((1 - f) * self.dominance + f * d_stim) * k + NEUTRE[2] * (1 - k)

    def deriver(self, dt):
        """Auto-decay : au fil du temps (dt secondes), l'humeur retombe vers le neutre (constante TAU_EMOTION)."""
        if dt <= 0:
            return
        k = math.exp(-dt / self.cfg.TAU_EMOTION)
        with self._verrou:
            self.valence *= k
            self.activation = NEUTRE[1] + (self.activation - NEUTRE[1]) * k
            self.dominance = NEUTRE[2] + (self.dominance - NEUTRE[2]) * k

    def apaiser(self, facteur=0.5):
        """Apaisement partiel (au réveil du sommeil) : l'humeur revient de moitié vers le neutre."""
        with self._verrou:
            self.valence *= facteur
            self.activation = self.activation * facteur + NEUTRE[1] * (1 - facteur)
            self.dominance = self.dominance * facteur + NEUTRE[2] * (1 - facteur)

    def retour_utilisateur(self, positif):
        """Effet d'un feedback sur l'humeur : 👍 la réjouit, 👎 l'assombrit."""
        signe = 1 if positif else -1
        with self._verrou:
            self.valence = max(-1.0, min(1.0, self.valence + signe * self.FEEDBACK_VALENCE))
            self.activation = max(0.0, min(1.0, self.activation + signe * self.FEEDBACK_ACTIVATION))

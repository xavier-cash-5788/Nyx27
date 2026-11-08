"""
nyx/config.py : configuration centrale de Nyx.

Sections :
  - Identité       : nom, utilisateur, chemins
  - Modèles        : modèles Ollama, timeout, réflexion préfrontale
  - Embeddings     : nomic-embed-text via Ollama
  - Limite mémoire : plafond de 2 Go par défaut, seuils et purge auto
  - Voix           : Piper (Siwis FR), moteur de synthèse léger
  - Rêves          : 2 rêves par sommeil, T dynamique 3 phases
  - Dissonance     : démon auto-actif qui résout les conflits internes
  - Archives       : notes compressées (.cpk), plafond 1 Go,
                     boucles autour des rêves (lecture avant, archivage après)
  - Mémoire        : decay, oubli, consolidation, distance de rappel
  - Émotions       : VAD, biais d'humeur
  - Pulsions       : fatigue, sommeil auto, pensées spontanées,
                     envie de lire (documents écrits par Nyx)
  - Exécution      : debug (non sauvegardé)

VERSION_FICHIER = 10 : ajout des paramètres de la pulsion "envie de lire".
"""
import json
import math
import os
from dataclasses import dataclass, field, fields

VERSION_FICHIER = 10
FICHIER_CONFIG_DEFAUT = "nyx_config.json"


def _champ(defaut, doc, groupe, mini=None, maxi=None, persistant=True, vide_ok=False):
    """Déclare un paramètre avec ses métadonnées."""
    return field(default=defaut, metadata={"doc": doc, "groupe": groupe, "mini": mini, "maxi": maxi,
                                           "persistant": persistant, "vide_ok": vide_ok})


@dataclass
class ConfigNyx:
    # ---------- identité et modèles ----------
    nom: str = _champ("Nyx", "Nom de l'agent", "Identité")
    utilisateur: str = _champ("Xavier", "Prénom de l'utilisateur", "Identité")
    model_name: str = _champ("qwen2.5:3b", "Modèle Ollama : conversation, journal, émotions", "Modèles")
    model_consolidation: str = _champ("", "Modèle Ollama pour la consolidation du sommeil (vide = même modèle)",
                                      "Modèles", vide_ok=True)
    num_ctx: int = _champ(8192, "Taille du contexte envoyée au modèle (tokens)", "Modèles", 512, 131072)
    reflexion_prefrontale: bool = _champ(True, "Plan interne avant chaque réponse", "Modèles")
    TIMEOUT_LLM: float = _champ(120.0, "Durée max d'un appel au modèle, en secondes.",
                                "Modèles", 0.0, 3600.0)
    chemin_db: str = _champ("./cerveau_memories", "Dossier de la base de mémoire (Chroma)", "Identité")
    ollama_keep_alive: str = _champ(
        "30s",
        "Durée pendant laquelle Ollama garde le LLM en VRAM après la dernière génération.",
        "Modèles",
    )

    # ---------- embeddings ----------
    embedding_model: str = _champ(
        "nomic-embed-text:latest",
        "Modèle Ollama pour les vecteurs de mémoire (embeddings)",
        "Embeddings",
    )
    embedding_base_url: str = _champ(
        "http://localhost:11434", "URL de l'API Ollama pour les embeddings", "Embeddings",
    )
    embedding_timeout: float = _champ(
        60.0, "Durée max d'un appel d'embedding, en secondes", "Embeddings", 1.0, 600.0,
    )
    embedding_cache_taille: int = _champ(
        2048, "Nombre de textes dont l'embedding est gardé en cache LRU", "Embeddings", 0, 100000,
    )

    # ---------- limite mémoire ----------
    memoire_limite_go: float = _champ(
        2.0, "Taille maximale du dossier de mémoire sur disque, en Go", "Limite mémoire", 0.1, 100.0,
    )
    memoire_seuil_avertissement: float = _champ(
        0.85, "Fraction de la limite à partir de laquelle Nyx avertit", "Limite mémoire", 0.1, 1.0,
    )
    memoire_purge_auto_actif: bool = _champ(
        True, "Purger automatiquement les souvenirs faibles quand la limite est dépassée",
        "Limite mémoire",
    )
    memoire_purge_cible: float = _champ(
        0.75, "Fraction de la limite à atteindre après une purge automatique",
        "Limite mémoire", 0.1, 1.0,
    )

    # ---------- voix (Piper seul) ----------
    voix_actif: bool = _champ(True, "Nyx parle à voix haute via Piper TTS", "Voix")
    voix_longueur: float = _champ(1.0, "Vitesse de la voix : >1 = plus lent, <1 = plus rapide", "Voix", 0.5, 2.0)
    voix_volume: float = _champ(1.0, "Volume de la voix", "Voix", 0.0, 2.0)
    voix_pensees: bool = _champ(True, "Nyx prononce aussi ses pensées spontanées", "Voix")
    voix_interpeller: bool = _champ(True, "Nyx peut t'interpeller spontanément", "Voix")
    voix_piper_modele: str = _champ("", "Chemin du modèle Piper (.onnx). Vide = chercher dans voix_piper_dossier.",
                                    "Voix Piper", vide_ok=True)
    voix_piper_dossier: str = _champ("models", "Dossier où chercher les modèles Piper", "Voix Piper")

    # ---------- rêves ----------
    reve_actif: bool = _champ(True, "Nyx rêve pendant son sommeil (2 rêves par sommeil)", "Rêves")
    reve_nb_fragments: int = _champ(5, "Nombre de fragments de souvenirs utilisés par rêve", "Rêves", 2, 20)
    reve_temperature_1: float = _champ(
        0.9, "Température du premier rêve (0.9 = créatif mais lisible)", "Rêves", 0.5, 1.5,
    )
    reve_temperature_2: float = _champ(
        1.0, "Température du second rêve (1.0 = surréaliste, délirant)", "Rêves", 0.5, 1.5,
    )
    reve_max_chars: int = _champ(
        2000, "Longueur max d'un rêve généré (caractères)", "Rêves", 200, 5000,
    )
    reve_seuil_souvenir: float = _champ(
        0.5, "Seuil de charge émotionnelle au-dessus duquel un rêve devient un souvenir onirique",
        "Rêves", 0.0, 1.0,
    )
    reve_raconter_spontanement: bool = _champ(
        True, "Nyx peut mentionner ses rêves spontanément si le rêve était fort", "Rêves",
    )

    # ---------- dissonance ----------
    dissonance_actif: bool = _champ(
        True, "Activer le démon qui détecte et résout les conflits internes", "Dissonance",
    )
    dissonance_seuil: float = _champ(
        0.6, "Seuil d'activation du démon (D > seuil → action)", "Dissonance", 0.1, 1.0,
    )
    dissonance_lambda: float = _champ(
        0.1, "Taux de fuite de la dissonance accumulée (par seconde)", "Dissonance", 0.01, 1.0,
    )
    dissonance_eta: float = _champ(
        0.1, "Taux d'apprentissage pour la mise à jour des poids", "Dissonance", 0.01, 1.0,
    )
    dissonance_budget_heure: int = _champ(
        3, "Nombre maximum d'actions du démon par heure", "Dissonance", 0, 20,
    )
    dissonance_intervalle_min: int = _champ(
        30, "Intervalle minimum entre deux actions du démon (secondes)", "Dissonance", 5, 3600,
    )
    dissonance_mode_simulation: bool = _champ(
        True,
        "Mode simulation : le démon observe et log mais n'agit pas (SÉCURITÉ)",
        "Dissonance",
    )

    # ---------- archives compressées ----------
    archives_dossier: str = _champ(
        "archives",
        "Dossier où stocker les notes archivées (format .cpk compressé)",
        "Archives",
    )
    archives_limite_go: float = _champ(
        1.0,
        "Taille maximale des archives compressées (en Go). Purge auto si dépassé.",
        "Archives", 0.1, 100.0,
    )
    archives_compresser_auto: bool = _champ(
        True,
        "Compresser automatiquement les notes ajoutées (format .cpk)",
        "Archives",
    )
    archives_reves_actif: bool = _champ(
        True,
        "Boucles d'archives autour des rêves : relecture aléatoire avant, archivage après",
        "Archives",
    )
    archives_reves_budget_heure: int = _champ(
        3,
        "Nombre max de rêves archivés automatiquement par heure",
        "Archives", 0, 50,
    )
    archives_reves_seuil_marquant: float = _champ(
        0.05,
        "Variation minimale de valence pour qu'un rêve soit 'marquant' et archivé",
        "Archives", 0.0, 1.0,
    )

    # ---------- mémoire ----------
    TAUX_DECAY: float = _champ(0.05, "Décroissance de la force d'un souvenir, par heure", "Mémoire", 0.0, 5.0)
    SEUIL_OUBLI: float = _champ(0.30, "Sous cette force, un souvenir est oublié au sommeil", "Mémoire", 0.0, 1.0)
    BONUS_REACTIVATION: float = _champ(0.15, "Force gagnée à chaque rappel d'un souvenir", "Mémoire", 0.0, 1.0)
    SEUIL_CHARGE_EMOTIONNELLE: float = _champ(0.5, "Au-dessus, un souvenir est « marquant »", "Mémoire", 0.0, 1.0)
    FACTEUR_DECAY_PROTEGE: float = _champ(0.3, "Les marquants et concepts oublient ce facteur de fois plus lentement",
                                          "Mémoire", 0.0, 1.0)
    SEUIL_DISTANCE: float = _champ(1.2, "Distance max (L2 Chroma) pour qu'un rappel soit pertinent",
                                   "Mémoire", 0.0, 4.0)
    MAX_HISTORIQUE: int = _champ(8, "Messages user/assistant gardés en mémoire de travail", "Mémoire", 2, 200)
    SEUIL_DOUBLON_JOURNAL: float = _champ(0.25, "Distance sous laquelle une réflexion du journal est redondante",
                                          "Mémoire", 0.0, 2.0)

    # ---------- émotions ----------
    POIDS_HUMEUR: float = _champ(0.30, "Force du biais d'humeur sur le rappel (0 = désactivé)",
                                 "Émotions", 0.0, 2.0)
    CANDIDATS_RAPPEL: int = _champ(8, "Souvenirs pertinents examinés avant le classement par humeur",
                                   "Émotions", 1, 50)
    TAU_EMOTION: int = _champ(900, "Constante de temps du retour de l'humeur au calme (secondes)",
                              "Émotions", 1, 86400)

    # ---------- pulsions ----------
    sommeil_auto_actif: bool = _champ(True, "Nyx s'endort d'elle-même quand elle est épuisée et inactive",
                                      "Pulsions")
    reflexion_auto_actif: bool = _champ(True, "Nyx pense à voix haute quand elle est inactive", "Pulsions")
    lecture_auto_actif: bool = _champ(True, "Nyx relit spontanément un de ses documents quand elle est inactive",
                                      "Pulsions")
    SEUIL_ENVIE_DORMIR: float = _champ(0.60, "Fatigue à partir de laquelle Nyx dit qu'elle a sommeil",
                                       "Pulsions", 0.0, 1.0)
    SEUIL_SOMMEIL_AUTO: float = _champ(0.85, "Fatigue à partir de laquelle Nyx s'endort si elle est inactive",
                                       "Pulsions", 0.0, 1.0)
    SEUIL_REFLEXION: float = _champ(0.60, "Envie de penser à partir de laquelle Nyx pense à voix haute",
                                    "Pulsions", 0.0, 1.0)
    SEUIL_ENVIE_LIRE: float = _champ(0.55, "Envie de lire à partir de laquelle Nyx relit un de ses documents",
                                     "Pulsions", 0.0, 1.0)
    SEUIL_EN_DECLIN: float = _champ(0.50, "Sous cette force (et au-dessus de l'oubli), un souvenir « s'estompe »",
                                    "Pulsions", 0.0, 1.0)
    FATIGUE_PAR_MESSAGE: float = _champ(0.05, "Fatigue ajoutée par échange", "Pulsions", 0.0, 1.0)
    FATIGUE_PAR_REFLEXION: float = _champ(0.02, "Fatigue ajoutée par pensée spontanée", "Pulsions", 0.0, 1.0)
    FATIGUE_PAR_MINUTE: float = _champ(0.004, "Fatigue ajoutée par minute d'éveil", "Pulsions", 0.0, 1.0)
    FATIGUE_MEMOIRE: float = _champ(0.30, "Poids des souvenirs presque oubliés dans la fatigue",
                                    "Pulsions", 0.0, 1.0)
    ENVIE_PAR_MINUTE: float = _champ(0.02, "Envie de penser gagnée par minute sans réflexion",
                                     "Pulsions", 0.0, 100.0)
    ENVIE_DECLIN: float = _champ(0.50, "Poids des souvenirs qui s'estompent dans l'envie de penser",
                                 "Pulsions", 0.0, 1.0)
    ENVIE_LIRE_PAR_MINUTE: float = _champ(0.03, "Envie de lire gagnée par minute sans lecture",
                                          "Pulsions", 0.0, 100.0)
    DELAI_INACTIVITE_SOMMEIL: int = _champ(45, "Secondes sans message avant un sommeil automatique",
                                           "Pulsions", 0, 86400)
    DELAI_INACTIVITE_REFLEXION: int = _champ(25, "Secondes sans message avant une pensée spontanée",
                                             "Pulsions", 0, 86400)
    DELAI_INACTIVITE_LECTURE: int = _champ(40, "Secondes sans message avant une lecture spontanée",
                                           "Pulsions", 0, 86400)
    DELAI_MIN_REFLEXION: int = _champ(180, "Secondes minimum entre deux pensées spontanées",
                                      "Pulsions", 0, 86400)
    DELAI_MIN_LECTURE: int = _champ(300, "Secondes minimum entre deux lectures spontanées",
                                    "Pulsions", 0, 86400)

    # ---------- exécution ----------
    debug: bool = _champ(False, "Affiche les distances de rappel dans la console",
                         "Exécution", persistant=False)

    # ------------------------------------------------------------------ construction

    @classmethod
    def defauts(cls):
        cfg = cls()
        cfg._finaliser()
        return cfg

    @classmethod
    def creer(cls, **surcharges):
        cfg = cls()
        for nom, valeur in surcharges.items():
            cfg.definir(nom, valeur)
        cfg._finaliser()
        return cfg

    def _finaliser(self):
        if not self.model_consolidation:
            self.model_consolidation = self.model_name

    # ------------------------------------------------------------------ validation

    @classmethod
    def _champ_par_nom(cls, nom):
        for f in fields(cls):
            if f.name == nom:
                return f
        raise KeyError(nom)

    @classmethod
    def valider(cls, nom, valeur):
        f = cls._champ_par_nom(nom)
        meta, attendu = f.metadata, f.type
        if attendu is bool:
            if not isinstance(valeur, bool):
                raise ValueError(f"{nom} : booléen attendu (reçu {valeur!r})")
            return valeur
        if attendu in (int, float):
            if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
                raise ValueError(f"{nom} : nombre attendu (reçu {valeur!r})")
            if isinstance(valeur, float) and not math.isfinite(valeur):
                raise ValueError(f"{nom} : valeur non finie ({valeur})")
            if attendu is int:
                if isinstance(valeur, float):
                    if not valeur.is_integer():
                        raise ValueError(f"{nom} : entier attendu (reçu {valeur})")
                    valeur = int(valeur)
            else:
                valeur = float(valeur)
            mini, maxi = meta["mini"], meta["maxi"]
            if (mini is not None and valeur < mini) or (maxi is not None and valeur > maxi):
                raise ValueError(f"{nom} : {valeur} hors bornes [{mini}, {maxi}]")
            return valeur
        if not isinstance(valeur, str):
            raise ValueError(f"{nom} : texte attendu (reçu {valeur!r})")
        valeur = valeur.strip()
        if not valeur and not meta["vide_ok"]:
            raise ValueError(f"{nom} : texte non vide attendu")
        return valeur

    def definir(self, nom, valeur):
        valeur = self.valider(nom, valeur)
        setattr(self, nom, valeur)
        self._finaliser()
        return valeur

    def coherence(self):
        pb = []
        if self.SEUIL_ENVIE_DORMIR >= self.SEUIL_SOMMEIL_AUTO:
            pb.append("SEUIL_ENVIE_DORMIR devrait être inférieur à SEUIL_SOMMEIL_AUTO.")
        if self.SEUIL_OUBLI >= self.SEUIL_EN_DECLIN:
            pb.append("SEUIL_OUBLI devrait être inférieur à SEUIL_EN_DECLIN.")
        if self.memoire_purge_cible >= self.memoire_seuil_avertissement:
            pb.append("memoire_purge_cible devrait être inférieure à memoire_seuil_avertissement.")
        return pb

    # ------------------------------------------------------------------ export, description

    def en_dict(self, persistants_seulement=True):
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.metadata["persistant"] or not persistants_seulement}

    @classmethod
    def champs_decrits(cls):
        defaut = cls.defauts()
        return [{"nom": f.name, "groupe": f.metadata["groupe"], "doc": f.metadata["doc"],
                 "type": f.type.__name__, "mini": f.metadata["mini"], "maxi": f.metadata["maxi"],
                 "defaut": getattr(defaut, f.name), "persistant": f.metadata["persistant"]}
                for f in fields(cls)]

    def appliquer(self, valeurs):
        problemes = []
        for nom, valeur in valeurs.items():
            try:
                self.definir(nom, valeur)
            except KeyError:
                problemes.append(f"paramètre inconnu ignoré : {nom}")
            except ValueError as e:
                problemes.append(f"valeur rejetée, défaut conservé — {e}")
        return problemes

    # ------------------------------------------------------------------ fichier

    @classmethod
    def depuis_fichier(cls, chemin=FICHIER_CONFIG_DEFAUT):
        cfg = cls.defauts()
        if not os.path.exists(chemin):
            return cfg, []
        try:
            with open(chemin, encoding="utf-8") as f:
                brut = json.load(f)
        except (OSError, ValueError) as e:
            return cfg, [f"fichier de configuration illisible ({e}) : valeurs par défaut utilisées"]
        if not isinstance(brut, dict):
            return cfg, ["fichier de configuration invalide (objet JSON attendu)"]
        avertissements = []
        version = brut.get("version")
        if isinstance(version, int) and version > VERSION_FICHIER:
            avertissements.append(f"fichier écrit par une version plus récente (v{version})")
        if isinstance(version, int) and version < VERSION_FICHIER:
            avertissements.append(f"mise à jour de la config v{version} → v{VERSION_FICHIER}")
        valeurs = brut.get("parametres", brut)
        valeurs = {k: v for k, v in valeurs.items() if k != "version"} if isinstance(valeurs, dict) else {}
        for f in fields(cls):
            if not f.metadata["persistant"] and f.name in valeurs:
                valeurs.pop(f.name)
                avertissements.append(f"{f.name} n'est pas sauvegardable : ignoré")
        avertissements += cfg.appliquer(valeurs)
        avertissements += cfg.coherence()
        return cfg, avertissements

    def sauver(self, chemin=FICHIER_CONFIG_DEFAUT):
        dossier = os.path.dirname(os.path.abspath(chemin))
        os.makedirs(dossier, exist_ok=True)
        temporaire = chemin + ".tmp"
        try:
            with open(temporaire, "w", encoding="utf-8") as f:
                json.dump({"version": VERSION_FICHIER, "parametres": self.en_dict()}, f,
                          ensure_ascii=False, indent=2)
            os.replace(temporaire, chemin)
        finally:
            if os.path.exists(temporaire):
                os.remove(temporaire)


CHAMPS = frozenset(f.name for f in fields(ConfigNyx))


class ProxyConfig:
    def __getattr__(self, nom):
        cfg = self.__dict__.get("cfg")
        if cfg is not None and nom in CHAMPS:
            return getattr(cfg, nom)
        raise AttributeError(f"{type(self).__name__!r} n'a pas d'attribut {nom!r}")

    def __setattr__(self, nom, valeur):
        cfg = self.__dict__.get("cfg")
        if cfg is not None and nom in CHAMPS:
            cfg.definir(nom, valeur)
        else:
            super().__setattr__(nom, valeur)

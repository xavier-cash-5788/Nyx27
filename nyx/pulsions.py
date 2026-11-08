"""
nyx/pulsions.py : l'horloge interne de Nyx (module 6/9).

Une pulsion = un thread qui fait `tick()` à intervalle régulier.
Un tick :
  1. dérive l'humeur (auto-decay émotionnel) ;
  2. recalcule fatigue, envie de réfléchir et envie de lire ;
  2b. émet les rappels d'agenda qui approchent ;
  3. déclenche AU PLUS UNE action spontanée :
       - "envie_dormir"    : annonce qu'elle a sommeil (une seule fois)
       - "dormir"          : s'endort automatiquement (si inactive + épuisée)
       - "auto_reflexion"  : pense à voix haute
       - "lecture_spontanee" : relit un de ses documents à voix haute

La production de la pensée est déléguée à un callback `penser` (injecté).
La lecture spontanée est déléguée à un callback `lire` (injecté).
Les rappels d'agenda sont délégués à un objet `agenda` (injecté).
La voix est déléguée à un callback `parler_voix` (injecté).

Rêves :
  - Le sommeil (automatique ou manuel) déclenche NREM + REM.
  - C'est le module Consolidation (injecté dans l'Agent) qui orchestre les rêves.
  - Le module Pulsions ne connaît PAS les rêves : il appelle juste `dormir()`.

Ce module ne connaît PAS Piper directement : il appelle juste le callback.
"""
import logging
import threading
import time

_log = logging.getLogger("nyx.pulsions")

PHRASES_SOMMEIL = [
    "Je commence à avoir sommeil... mes pensées se font plus lentes.",
    "J'ai les idées qui s'embrouillent. J'aurais besoin de dormir bientôt.",
    "Un peu de fatigue se fait sentir : un sommeil m'aiderait à trier mes souvenirs.",
]

PHRASES_LECTURE = [
    "J'ai envie de relire un de mes écrits, pour voir où j'en étais.",
    "Mes documents me manquent un peu, je vais en relire un.",
    "Envie de replonger dans une de mes notes, histoire de renouer avec moi-même.",
]


class Pulsions:
    def __init__(self, cfg, memoire, limbique, horloge=None,
                 dormir=None, penser=None, sur_evenement=None, agenda=None,
                 parler_voix=None, lire=None):
        """
        cfg             ConfigNyx (tous les seuils relus à chaque tick)
        memoire         Memoire (pour en_declin() et renforcer())
        limbique        Limbique (pour deriver(dt))
        horloge         fonction sans argument retournant time.time() (injectable)
        dormir          callback() appelé pour endormir Nyx (injecté)
        penser          callback(cibles) -> texte ou None (injecté)
        sur_evenement   callback(genre, texte) branché par l'interface
        agenda          Agenda optionnel : si fourni, ses rappels sont émis
        parler_voix     callback(texte) -> bool : fait prononcer le texte à voix haute
        lire            callback() -> dict ou None : relit un document spontanément
        """
        self.cfg = cfg
        self.memoire = memoire
        self.limbique = limbique
        self._maintenant = horloge or time.time
        self._dormir = dormir or (lambda: None)
        self._penser = penser or (lambda cibles: None)
        self._lire = lire or (lambda: None)
        self.sur_evenement = sur_evenement
        self.agenda = agenda
        self._parler_voix = parler_voix or (lambda texte: False)

        # État interne
        self.debut_eveil = self._maintenant()
        self.derniere_interaction = self.debut_eveil
        self.derniere_auto_reflexion = self.debut_eveil
        self.derniere_lecture = self.debut_eveil
        self.dernier_tick = self.debut_eveil
        self.derniere_pensee = None          # (epoch, texte)
        self.derniere_lecture_texte = None   # (epoch, titre, extrait)
        self.envie_dormir_annoncee = False
        self.charge_msgs = 0.0
        self.fatigue_courante = 0.0
        self.pulsions_cache = {}

        # Thread
        self._verrou = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ lecture
    def noter_interaction(self, maintenant=None):
        self.derniere_interaction = maintenant if maintenant is not None else self._maintenant()

    def ajouter_charge_message(self, quantite=None):
        q = quantite if quantite is not None else self.cfg.FATIGUE_PAR_MESSAGE
        self.charge_msgs = min(1.0, self.charge_msgs + q)

    def ajouter_charge_reflexion(self):
        self.charge_msgs = min(1.0, self.charge_msgs + self.cfg.FATIGUE_PAR_REFLEXION)

    def reset_apres_sommeil(self, maintenant=None):
        """Après le sommeil : fatigue à 0, réinitialisation des compteurs."""
        m = maintenant if maintenant is not None else self._maintenant()
        self.charge_msgs = 0.0
        self.debut_eveil = m
        self.derniere_auto_reflexion = m
        self.derniere_lecture = m
        self.derniere_interaction = m
        self.dernier_tick = m
        self.envie_dormir_annoncee = False

    # ------------------------------------------------------------------ calcul
    def calculer(self, maintenant=None):
        """
        Fatigue = messages échangés + temps éveillé + encombrement.
        Envie de réfléchir = temps sans penser + part de souvenirs en déclin.
        Envie de lire = temps sans lire + nombre de documents disponibles.
        """
        m = maintenant if maintenant is not None else self._maintenant()
        declin, faibles, total = self.memoire.en_declin()
        part_faibles = faibles / total if total else 0.0
        part_declin = (len(declin) + faibles) / total if total else 0.0
        minutes_eveil = max(0.0, (m - self.debut_eveil) / 60)
        minutes_sans_pensee = max(0.0, (m - self.derniere_auto_reflexion) / 60)
        minutes_sans_lecture = max(0.0, (m - self.derniere_lecture) / 60)

        fatigue = min(1.0,
                      self.charge_msgs
                      + minutes_eveil * self.cfg.FATIGUE_PAR_MINUTE
                      + self.cfg.FATIGUE_MEMOIRE * part_faibles)
        envie = min(1.0,
                    minutes_sans_pensee * self.cfg.ENVIE_PAR_MINUTE
                    + self.cfg.ENVIE_DECLIN * part_declin)
        envie_lire = min(1.0,
                         minutes_sans_lecture * self.cfg.ENVIE_LIRE_PAR_MINUTE)

        self.fatigue_courante = fatigue
        self.pulsions_cache = {
            "fatigue": fatigue,
            "envie_reflexion": envie,
            "envie_lire": envie_lire,
            "nb_declin": len(declin), "nb_faibles": faibles, "total": total,
            "part_declin": part_declin,
            "inactivite": max(0.0, m - self.derniere_interaction),
        }
        return dict(self.pulsions_cache)

    # ------------------------------------------------------------------ parole

    def _parler(self, genre, texte, prononcer=True):
        """
        Émet un événement vers l'interface et, si `prononcer` est vrai ET que
        la voix est active ET que les pensées à voix haute sont activées,
        fait prononcer le texte par Nyx (Piper).
        """
        if self.sur_evenement:
            try:
                self.sur_evenement(genre, texte)
            except Exception as e:
                _log.warning("Callback d'interface : %s", e)

        if not prononcer:
            return
        if not getattr(self.cfg, "voix_actif", False):
            return
        if not getattr(self.cfg, "voix_pensees", False):
            return
        texte_vocal = self._texte_vocal(texte)
        try:
            self._parler_voix(texte_vocal)
        except Exception as e:
            _log.debug("Voix pulsion : %s", e)

    @staticmethod
    def _texte_vocal(texte):
        """
        Prépare un texte pour la prononciation :
          - retire les emojis / marqueurs visuels au début ;
          - tronque à la première phrase si c'est trop long.
        """
        if not texte:
            return ""
        t = texte.lstrip()
        while t and not (t[0].isalnum() or t[0] in "«\"'("):
            t = t[1:].lstrip()
        if len(t) > 400:
            for i, c in enumerate(t):
                if i >= 200 and c in ".!?":
                    return t[:i + 1]
            return t[:400]
        return t

    def _phrase_fatigue(self, fatigue):
        if fatigue >= self.cfg.SEUIL_SOMMEIL_AUTO:
            return "Je n'en peux plus : je dois dormir, maintenant."
        return PHRASES_SOMMEIL[int(fatigue * 100) % len(PHRASES_SOMMEIL)]

    def _phrase_lecture(self, envie):
        return PHRASES_LECTURE[int(envie * 100) % len(PHRASES_LECTURE)]

    # ------------------------------------------------------------------ agenda

    def _emettre_rappels_agenda(self, m):
        """Émet les rappels d'agenda dus à cet instant (chacun une seule fois)."""
        if self.agenda is None:
            return
        try:
            a_rappeler = self.agenda.a_rappeler(m)
        except Exception as e:
            _log.warning("Lecture des rappels agenda impossible : %s", e)
            return
        for ev in a_rappeler:
            try:
                minutes = int(ev.minutes_avant(m))
                quand = f"dans {minutes} min" if minutes > 0 else "maintenant"
                self._parler("rappel", f"{ev.titre} — {quand}")
                self.agenda.marquer_rappele(ev.id)
            except Exception as e:
                _log.warning("Émission du rappel %r impossible : %s",
                             getattr(ev, "id", "?"), e)

    # ------------------------------------------------------------------ tick
    def tick(self, maintenant=None):
        """
        Un battement de l'horloge interne. Retourne "envie_dormir", "dormir",
        "auto_reflexion", "lecture_spontanee" ou None.
        """
        if not self._verrou.acquire(blocking=False):
            return None  # Nyx est occupée
        try:
            m = maintenant if maintenant is not None else self._maintenant()
            dt = max(0.0, m - self.dernier_tick)
            self.dernier_tick = m

            # 1. Auto-decay émotionnel
            self.limbique.deriver(dt)

            # 2. Recalcul pulsions
            p = self.calculer(m)
            fatigue, inactif = p["fatigue"], p["inactivite"]
            envie, envie_lire = p["envie_reflexion"], p["envie_lire"]

            # 2b. Rappels d'agenda
            self._emettre_rappels_agenda(m)

            # 3. Au plus UNE action spontanée (par ordre de priorité)
            if (self.cfg.sommeil_auto_actif
                    and fatigue >= self.cfg.SEUIL_SOMMEIL_AUTO
                    and inactif >= self.cfg.DELAI_INACTIVITE_SOMMEIL):
                self._parler("sommeil", "Je m'endors pour trier mes souvenirs...")
                self._dormir()
                self._parler("reveil", "Je me réveille, l'esprit plus clair.")
                return "dormir"

            if fatigue >= self.cfg.SEUIL_ENVIE_DORMIR and not self.envie_dormir_annoncee:
                self.envie_dormir_annoncee = True
                self._parler("envie_dormir", self._phrase_fatigue(fatigue))
                return "envie_dormir"

            peut_penser = not (self.cfg.sommeil_auto_actif
                               and fatigue >= self.cfg.SEUIL_SOMMEIL_AUTO)

            # Envie de penser (priorité légèrement supérieure à la lecture)
            if (self.cfg.reflexion_auto_actif and peut_penser
                    and envie >= self.cfg.SEUIL_REFLEXION
                    and inactif >= self.cfg.DELAI_INACTIVITE_REFLEXION
                    and m - self.derniere_auto_reflexion >= self.cfg.DELAI_MIN_REFLEXION):
                if self.auto_reflexion(force=False, maintenant=m):
                    return "auto_reflexion"

            # Envie de lire (déclenchée si pas de pensée en cours)
            if (getattr(self.cfg, "lecture_auto_actif", True) and peut_penser
                    and envie_lire >= self.cfg.SEUIL_ENVIE_LIRE
                    and inactif >= self.cfg.DELAI_INACTIVITE_LECTURE
                    and m - self.derniere_lecture >= self.cfg.DELAI_MIN_LECTURE):
                if self.lecture_spontanee(maintenant=m):
                    return "lecture_spontanee"

            return None
        finally:
            self._verrou.release()

    # ------------------------------------------------------------------ pensée à voix haute
    def auto_reflexion(self, force=False, maintenant=None):
        """
        Nyx pense à voix haute. La production du texte est déléguée au callback `penser`.
        `force=True` pense même sans souvenir en déclin.
        Retourne le texte de la pensée, ou None.
        """
        m = maintenant if maintenant is not None else self._maintenant()
        with self._verrou:
            self.derniere_auto_reflexion = m

            declin, _, _ = self.memoire.en_declin()
            cibles = declin[:2]
            if not cibles:
                if not force:
                    return None
                recents = self.memoire.tous(cache_s=0.0)
                recents.sort(key=lambda s: s.created_at, reverse=True)
                cibles = recents[:2]

            pensee = self._penser(cibles)
            if not pensee:
                return None

            ids_cibles = [s.id for s in cibles]
            self.memoire.renforcer(ids_cibles)

            self.derniere_pensee = (m, pensee)
            self.ajouter_charge_reflexion()
            self.calculer(m)
            self._parler("auto_reflexion", pensee)
            return pensee

    # ------------------------------------------------------------------ lecture spontanée
    def lecture_spontanee(self, maintenant=None):
        """
        Nyx relit un de ses documents à voix haute.
        Le choix du document + le texte à lire sont délégués au callback `lire`.
        Retourne un dict {"titre":..., "extrait":...} ou None.
        """
        m = maintenant if maintenant is not None else self._maintenant()
        with self._verrou:
            self.derniere_lecture = m

            try:
                resultat = self._lire()
            except Exception as e:
                _log.warning("Lecture spontanée échouée : %s", e)
                return None

            if not resultat:
                return None

            titre = resultat.get("titre", "?")
            extrait = resultat.get("extrait", "")
            if not extrait:
                return None

            # Annonce + lecture
            self._parler(
                "lecture_spontanee",
                f"{self._phrase_lecture(self.calculer(m)['envie_lire'])} "
                f"Je relis « {titre} » :\n\n{extrait}",
            )
            self.derniere_lecture_texte = (m, titre, extrait)
            return {"titre": titre, "extrait": extrait}

    # ------------------------------------------------------------------ thread
    def demarrer(self, intervalle=5.0):
        if self._thread and self._thread.is_alive():
            return

        def boucle():
            while not self._stop.wait(intervalle):
                try:
                    self.tick()
                except Exception as e:
                    _log.warning("Erreur de l'horloge interne : %s", e)

        self._stop.clear()
        self._thread = threading.Thread(target=boucle, daemon=True, name="nyx-pulsions")
        self._thread.start()

    def arreter(self):
        self._stop.set()

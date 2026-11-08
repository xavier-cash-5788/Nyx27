"""
nyx/dissonance.py : le Démon de Dissonance de Nyx.

Inspiré des neurosciences de la dissonance cognitive :
  - Conflict monitoring (Botvinick) : l'ACC et l'insula détectent les
    conflits entre représentations concurrentes.
  - Erreur de prédiction (Schultz) : la dopamine code l'écart entre
    attendu et obtenu.
  - Free Energy Principle (Friston) : la dissonance est une surprise qu'on
    réduit en changeant son modèle (croyance) ou en agissant (active inference).
  - Résolution : le cortex préfrontal arbitre.

Modèle mathématique :
  1. Dissonance = énergie d'un réseau de contraintes
       E = -0.5 × Σ w_ij · s_i · s_j
     où w_ij > 0 pour cohérent, w_ij < 0 pour contradictoire.
  2. Conflit instantané (Botvinick) : C = a_1 × a_2 (max quand deux
     réponses sont fortes simultanément).
  3. Accumulation à fuite :
       dD/dt = -λ·D + conflit(t)
     Le démon agit seulement quand D > θ.
  4. Résolution par descente de gradient :
       s_i ← s_i - η · ∂E/∂s_i
  5. Apprentissage (règle de Hebb) : mise à jour de w_ij pour éviter que
     le même conflit ne revienne.

GARDE-FOUS (obligatoires) :
  - Limite de fréquence (1 action par cycle min)
  - Budget d'actions par heure
  - Mode simulation (aucune action réelle, juste log)
  - Interrupteur d'arrêt immédiat
  - Journalisation de chaque action

Sans ces garde-fous, un processus qui réduit sa propre dissonance peut
se « rationaliser » et dériver — exactement comme chez l'humain.

V2 : propriétés `seuil` et `mode_simulation` ajoutées pour compatibilité
     avec l'interface graphique (nyx_gui.py).
"""
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

_log = logging.getLogger("nyx.dissonance")


# ==================================================================== dataclasses

@dataclass
class Conflit:
    """Un conflit détecté entre deux éléments internes."""
    id: str
    timestamp: str
    element_a: str        # Texte du premier élément
    element_b: str        # Texte du second élément
    similarite: float     # Distance sémantique (0 = identiques, 1 = très différents)
    energie: float        # Énergie de dissonance E
    conflit: float        # Conflit instantané C
    valence_a: float = 0.0
    valence_b: float = 0.0
    type_conflit: str = "inconnu"  # "croyance_croyance", "croyance_action", "objectif_objectif"
    resolu: bool = False
    action_prise: str = ""
    gain_energie: float = 0.0


@dataclass
class EtatDemon:
    """État interne du démon."""
    dissonance_actuelle: float = 0.0        # D accumulé
    seuil: float = 0.6                     # θ
    conflits_actifs: List[str] = field(default_factory=list)
    derniere_action: float = 0.0
    actions_heure: int = 0
    heure_courante: int = 0
    actif: bool = True
    en_pause: bool = False
    mode_simulation: bool = True           # Par défaut : simulation !
    dernier_conflit: Optional[Conflit] = None
    historique_resolutions: List[dict] = field(default_factory=list)

    def en_dict(self):
        d = asdict(self)
        d["dernier_conflit"] = asdict(self.dernier_conflit) if self.dernier_conflit else None
        return d


# ==================================================================== démon

class DemonDissonance:
    """
    Démon auto-actif qui détecte et résout les dissonances internes de Nyx.

    Utilisation :
        demon = DemonDissonance(cfg, memoire, journal, llm, sur_evenement=...)
        demon.demarrer(intervalle=10.0)
        demon.arreter()
        demon.mode_simulation = False  # pour activer les vraies résolutions
    """

    def __init__(self, cfg, memoire, journal, llm,
                 sur_evenement=None, collection_conflits=None):
        """
        cfg                  ConfigNyx
        memoire              Memoire
        journal              Journal
        llm                  ClientLLM
        sur_evenement        callback(genre, texte) pour l'interface
        collection_conflits  collection Chroma pour stocker les conflits
                             (créée si None)
        """
        self.cfg = cfg
        self.memoire = memoire
        self.journal = journal
        self.llm = llm
        self.sur_evenement = sur_evenement
        self.collection = collection_conflits

        self.etat = EtatDemon(
            seuil=cfg.dissonance_seuil,
            mode_simulation=cfg.dissonance_mode_simulation,
        )

        self._verrou = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

        # Paramètres du modèle
        self.theta = cfg.dissonance_seuil
        self.lambda_decay = cfg.dissonance_lambda  # fuite
        self.eta = cfg.dissonance_eta  # taux d'apprentissage
        self.budget_heure = cfg.dissonance_budget_heure
        self.intervalle_min_action = cfg.dissonance_intervalle_min

    # ---------------------------------------------------------------- propriétés (compat GUI)

    @property
    def seuil(self):
        """Seuil d'activation θ (alias vers etat.seuil)."""
        return self.etat.seuil

    @seuil.setter
    def seuil(self, valeur):
        self.etat.seuil = float(valeur)
        self.theta = float(valeur)

    @property
    def mode_simulation(self):
        """Mode simulation (alias vers etat.mode_simulation)."""
        return self.etat.mode_simulation

    @mode_simulation.setter
    def mode_simulation(self, valeur):
        self.etat.mode_simulation = bool(valeur)

    # ---------------------------------------------------------------- cycle

    def demarrer(self, intervalle=10.0):
        """Lance le démon en arrière-plan."""
        if self._thread and self._thread.is_alive():
            return

        def boucle():
            _log.info("🧠 Démon de Dissonance démarré (intervalle %.1fs, simulation=%s)",
                      intervalle, self.etat.mode_simulation)
            while not self._stop.wait(intervalle):
                try:
                    self.tick()
                except Exception as e:
                    _log.warning("Erreur du démon : %s", e)

        self._stop.clear()
        self._thread = threading.Thread(target=boucle, daemon=True, name="nyx-demon-dissonance")
        self._thread.start()

    def arreter(self):
        """Arrête le démon proprement."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ---------------------------------------------------------------- tick

    def tick(self, maintenant=None):
        """
        Un cycle du démon :
          1. Observe : cherche les conflits potentiels
          2. Accumule la dissonance
          3. Si D > θ ET budget disponible : agit
        """
        with self._verrou:
            m = time.time() if maintenant is None else maintenant

            # Garde-fou : interrupteur
            if not self.etat.actif or self.etat.en_pause:
                return None

            # Garde-fou : fréquence minimale
            if m - self.etat.derniere_action < self.intervalle_min_action:
                return None

            # Garde-fou : budget horaire
            heure = int(m // 3600)
            if heure != self.etat.heure_courante:
                self.etat.heure_courante = heure
                self.etat.actions_heure = 0
            if self.etat.actions_heure >= self.budget_heure:
                _log.info("🧠 Budget horaire atteint (%d/%d), démon en pause",
                          self.etat.actions_heure, self.budget_heure)
                return None

            # 1. Détection des conflits
            conflits = self._detecter_conflits()
            self.etat.conflits_actifs = [c.id for c in conflits]

            # 2. Accumulation avec fuite
            conflit_total = sum(c.conflit for c in conflits)
            dt = max(0.1, m - (self.etat.derniere_action or m))
            self.etat.dissonance_actuelle = max(
                0.0,
                self.etat.dissonance_actuelle * math.exp(-self.lambda_decay * dt) + conflit_total * dt
            )

            # 3. Décision
            if self.etat.dissonance_actuelle < self.etat.seuil:
                return None  # pas de réaction au bruit

            if not conflits:
                return None  # D élevé mais aucun conflit actuel : on laisse retomber

            # 4. Choix de l'action optimale (minimise E pour le moindre coût)
            meilleur = self._choisir_action(conflits)

            # 5. Exécution
            self._executer_action(meilleur, m)
            return meilleur

    # ---------------------------------------------------------------- détection

    def _detecter_conflits(self) -> List[Conflit]:
        """
        Détecte les conflits internes :
          - paires de souvenirs de valence opposée sur le même sujet
          - croyances contradictoires (via LLM sur échantillon)
        """
        conflits = []
        souvenirs = self.memoire.tous(cache_s=30.0)
        if len(souvenirs) < 2:
            return conflits

        # Échantillon aléatoire pour ne pas tout traiter à chaque tick
        import random
        if len(souvenirs) > 50:
            echantillon = random.sample(souvenirs, min(20, len(souvenirs)))
        else:
            echantillon = souvenirs

        # Paires de souvenirs de valence opposée, sémantiquement proches
        for i, a in enumerate(echantillon):
            for b in echantillon[i+1:]:
                # Distance sémantique
                dist_sem = self._distance_semantique(a.texte, b.texte)
                if dist_sem > 0.4:  # pas assez proche pour être un conflit
                    continue

                # Conflit émotionnel
                conflit_emo = abs(a.valence - b.valence)
                if conflit_emo < 0.5:  # pas assez de conflit
                    continue

                # Énergie de dissonance
                energie = -0.5 * (-1.0) * a.valence * b.valence  # w = -1 si contradictoire
                conflit_instant = a.activation * b.activation

                if energie < 0.1:
                    continue

                # Crée le conflit
                c = Conflit(
                    id=str(uuid.uuid4())[:12],
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    element_a=a.texte[:200],
                    element_b=b.texte[:200],
                    similarite=dist_sem,
                    energie=energie,
                    conflit=conflit_instant,
                    valence_a=a.valence,
                    valence_b=b.valence,
                    type_conflit="croyance_croyance",
                )
                conflits.append(c)

        return conflits[:5]  # garde les 5 plus importants

    def _distance_semantique(self, texte_a, texte_b):
        """Distance sémantique via Chroma (0 = identiques, 2 = opposés)."""
        try:
            if self.memoire.ef is not None:
                emb_a = self.memoire.ef.embed_query(texte_a)
                emb_b = self.memoire.ef.embed_query(texte_b)
                # Distance cosinus
                dot = sum(x * y for x, y in zip(emb_a, emb_b))
                n_a = math.sqrt(sum(x * x for x in emb_a))
                n_b = math.sqrt(sum(x * x for x in emb_b))
                if n_a == 0 or n_b == 0:
                    return 1.0
                return 1.0 - (dot / (n_a * n_b))
        except Exception as e:
            _log.debug("Erreur distance sémantique : %s", e)
        return 1.0

    # ---------------------------------------------------------------- action

    def _choisir_action(self, conflits: List[Conflit]) -> Conflit:
        """
        Choisit le conflit à résoudre en minimisant (energie × coût).
        Le coût varie selon le levier choisi (voir _resoudre).
        """
        # Pour simplifier : on prend le conflit avec la plus grande énergie
        return max(conflits, key=lambda c: c.energie)

    def _executer_action(self, conflit: Conflit, maintenant: float):
        """
        Exécute la résolution (ou simulation selon le mode).
        """
        # Construit la résolution
        resolution = self._resoudre(conflit)

        if self.etat.mode_simulation:
            _log.info("🧠 [SIMULATION] Conflit %s → %s", conflit.id, resolution["description"])
            conflit.action_prise = f"[simulation] {resolution['description']}"
            self._notifier("dissonance_simulation",
                           f"Simulation : {resolution['description']}")
        else:
            _log.info("🧠 [ACTION] Conflit %s → %s", conflit.id, resolution["description"])
            resolution["executer"]()
            conflit.action_prise = resolution["description"]
            conflit.gain_energie = resolution["gain_energie"]
            conflit.resolu = True
            self._notifier("dissonance_resolue",
                           f"Résolu : {resolution['description']}")

        # Mise à jour de l'état
        self.etat.derniere_action = maintenant
        self.etat.actions_heure += 1
        self.etat.dissonance_actuelle = 0.0  # reset après action
        self.etat.dernier_conflit = conflit
        self.etat.historique_resolutions.append({
            "timestamp": conflit.timestamp,
            "conflit_id": conflit.id,
            "action": conflit.action_prise,
            "gain_energie": conflit.gain_energie,
            "mode": "simulation" if self.etat.mode_simulation else "action",
        })
        # Garde seulement les 50 dernières
        self.etat.historique_resolutions = self.etat.historique_resolutions[-50:]

        # Écrit au journal
        try:
            self.journal.ecrire(
                contexte=f"[DISSONANCE {conflit.id}] {resolution['description']}",
                type_evenement="dissonance",
                prompt_identite="Tu es Nyx, un agent IA.",
                emotion="Neutralité", emoji="🧠",
                vad=(0.0, 0.5, 0.5),
            )
        except Exception as e:
            _log.debug("Erreur journal dissonance : %s", e)

    def _resoudre(self, conflit: Conflit) -> dict:
        """
        Choisit le levier de résolution qui minimise E pour le moindre coût.

        4 leviers (Festinger) :
          1. Changer la croyance A (coût : bas, impact : moyen)
          2. Changer la croyance B (coût : bas, impact : moyen)
          3. Ajouter une cognition consonante (coût : moyen, impact : élevé)
          4. Diminuer l'importance du conflit (coût : très bas, impact : bas)

        Le démon choisit celui qui baisse le plus E / coût.
        """
        options = []

        # Option 1 : changer A
        cout_1 = 0.2
        gain_1 = conflit.energie * 0.5
        options.append({
            "description": f"Changer la croyance A (valence {conflit.valence_a:+.1f} → neutre)",
            "gain_energie": gain_1,
            "cout": cout_1,
            "ratio": gain_1 / cout_1,
            "executer": lambda: self._adoucir_valence(
                conflit.element_a, conflit.valence_a, "adoucir A"
            ),
        })

        # Option 2 : changer B
        cout_2 = 0.2
        gain_2 = conflit.energie * 0.5
        options.append({
            "description": f"Changer la croyance B (valence {conflit.valence_b:+.1f} → neutre)",
            "gain_energie": gain_2,
            "cout": cout_2,
            "ratio": gain_2 / cout_2,
            "executer": lambda: self._adoucir_valence(
                conflit.element_b, conflit.valence_b, "adoucir B"
            ),
        })

        # Option 3 : ajouter une cognition consonante
        cout_3 = 0.4
        gain_3 = conflit.energie * 0.8
        options.append({
            "description": f"Créer un concept de synthèse qui réconcilie A et B",
            "gain_energie": gain_3,
            "cout": cout_3,
            "ratio": gain_3 / cout_3,
            "executer": lambda: self._synthetiser(conflit),
        })

        # Option 4 : diminuer l'importance (oubli accéléré)
        cout_4 = 0.1
        gain_4 = conflit.energie * 0.3
        options.append({
            "description": f"Réduire la force des deux souvenirs (oubli accéléré)",
            "gain_energie": gain_4,
            "cout": cout_4,
            "ratio": gain_4 / cout_4,
            "executer": lambda: self._reduire_force(conflit),
        })

        # Choisir le meilleur ratio
        meilleur = max(options, key=lambda o: o["ratio"])
        return meilleur

    def _adoucir_valence(self, texte, valence, label):
        """Adoucit la valence d'un souvenir (la rapproche de 0)."""
        souvenirs = self.memoire.tous()
        for s in souvenirs:
            if texte[:80] in s.texte:
                data = self.memoire.col.get(ids=[s.id])
                if data["ids"]:
                    meta = data["metadatas"][0]
                    meta["valence"] = meta.get("valence", 0.0) * 0.7
                    self.memoire.col.update(ids=[s.id], metadatas=[meta])
                    _log.info("🧠 %s : valence %.2f → %.2f",
                              label, s.valence, meta["valence"])
                    break

    def _synthetiser(self, conflit: Conflit):
        """Demande au LLM un concept qui réconcilie A et B."""
        prompt = f"""Deux souvenirs de Nyx se contredisent. Écris UN concept de synthèse
qui réconcilie les deux en une seule phrase (max 200 caractères).

Souvenir A : {conflit.element_a}
Souvenir B : {conflit.element_b}

Réponds UNIQUEMENT avec le concept, sans introduction."""
        try:
            rep = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.5)
            concept = rep.texte.strip()
            if concept and 20 <= len(concept) <= 250:
                self.memoire.ajouter(
                    texte=f"[Synthèse] {concept}",
                    valence=0.0, activation=0.5,
                    type="concept_consolidé", force=0.8,
                )
                _log.info("🧠 Concept de synthèse créé : %s", concept[:80])
        except Exception as e:
            _log.warning("Erreur synthèse : %s", e)

    def _reduire_force(self, conflit: Conflit):
        """Réduit la force des deux souvenirs concernés."""
        for texte in (conflit.element_a, conflit.element_b):
            souvenirs = self.memoire.tous()
            for s in souvenirs:
                if texte[:80] in s.texte:
                    data = self.memoire.col.get(ids=[s.id])
                    if data["ids"]:
                        meta = data["metadatas"][0]
                        meta["force_base"] = max(0.1, meta.get("force_base", 0.5) * 0.8)
                        self.memoire.col.update(ids=[s.id], metadatas=[meta])
                        break

    # ---------------------------------------------------------------- contrôle

    def pause(self):
        self.etat.en_pause = True

    def reprendre(self):
        self.etat.en_pause = False

    def activer(self):
        self.etat.actif = True

    def desactiver(self):
        self.etat.actif = False

    def toggle_simulation(self):
        self.etat.mode_simulation = not self.etat.mode_simulation
        _log.info("🧠 Mode simulation : %s", self.etat.mode_simulation)
        return self.etat.mode_simulation

    def _notifier(self, genre, texte):
        if self.sur_evenement:
            try:
                self.sur_evenement(genre, texte)
            except Exception as e:
                _log.debug("Callback : %s", e)

    def etat_actuel(self):
        return self.etat.en_dict()

    def diagnostique(self):
        """Retourne un résumé lisible."""
        e = self.etat
        return (
            f"🧠 Démon de Dissonance\n"
            f"   Actif          : {e.actif}\n"
            f"   En pause       : {e.en_pause}\n"
            f"   Simulation     : {e.mode_simulation}\n"
            f"   Dissonance D   : {e.dissonance_actuelle:.3f} / seuil {e.seuil}\n"
            f"   Conflits actifs: {len(e.conflits_actifs)}\n"
            f"   Actions cette h: {e.actions_heure} / budget {self.budget_heure}\n"
            f"   Résolutions    : {len(e.historique_resolutions)}\n"
        )

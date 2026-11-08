"""
nyx/boucles_archives.py : boucles d'archivage et de lecture autour des rêves.

Deux mécanismes :

  1. LECTURE ALÉATOIRE AVANT LE RÊVE
     Avant chaque cycle REM, Nyx tire une note d'archive au hasard
     et la relit. Le texte est ensuite injecté dans le rêve comme
     fragment prioritaire (préfixe "📖 Note relue : …").
     → Donne une continuité narrative aux rêves.

  2. ARCHIVAGE APRÈS LE RÊVE
     Après le cycle REM, si au moins un rêve a été produit ET qu'il
     est "marquant" (charge émotionnelle ≥ seuil), Nyx archive
     automatiquement une note type "reve".
     → Laisse une trace des rêves dans les archives.

Garde-fous :
  - Budget horaire (par défaut 3 archives de rêve/heure).
  - Flag cfg.archives_reves_actif (désactivable depuis les réglages).
  - Erreurs silencieuses : ne casse jamais le sommeil.
  - Anti-doublon : ne réarchive pas le même rêve (par timestamp).
"""
import logging
import random
import threading
import time

_log = logging.getLogger("nyx.boucles_archives")

# Budget : nombre max d'archives de rêve par heure
BUDGET_DEFAUT = 3


class BouclesArchives:
    """
    Orchestration des deux boucles autour des rêves.

    Utilisation dans Consolidation :
        self.boucles = BouclesArchives(cfg, archives)
        # Avant le cycle REM :
        note_lue = self.boucles.avant_reve()
        # Après le cycle REM :
        self.boucles.apres_reve(liste_reves, nom="Nyx")

    Paramètres :
        cfg       ConfigNyx
        archives  instance de nyx.archives.Archives
        sur_evenement  callback(genre, texte) optionnel
        budget_heure   nombre max d'archives de rêve par heure
    """

    def __init__(self, cfg, archives, sur_evenement=None,
                 budget_heure=BUDGET_DEFAUT):
        self.cfg = cfg
        self.archives = archives
        self.sur_evenement = sur_evenement
        self.budget_heure = int(budget_heure)

        self._verrou = threading.RLock()
        self._derniere_note_lue = None       # {"id":..., "titre":...}
        self._derniers_reves_archives = set()  # timestamps des rêves déjà archivés

        # Compteur d'archives de rêve pour l'heure courante
        self._heure_courante = int(time.time() // 3600)
        self._archives_heure = 0

    # ---------------------------------------------------------------- lecture

    def avant_reve(self):
        """
        Tire une note au hasard et la retourne sous forme de texte
        (ou None si aucune note, ou si la boucle est désactivée).
        N'échoue jamais : le rêve doit pouvoir se faire même sans note.
        """
        if not getattr(self.cfg, "archives_reves_actif", True):
            return None
        try:
            notes = self.archives.lister(tri="date_desc")
            if not notes:
                return None

            # Tirage aléatoire parmi les notes
            note = random.choice(notes)
            contenu = self.archives.lire_note(note["id"])
            if contenu is None:
                _log.warning("BouclesArchives : note introuvable (%s)", note["id"])
                return None

            # Tronque pour ne pas noyer le prompt du rêve
            contenu_court = contenu.strip()
            if len(contenu_court) > 400:
                contenu_court = contenu_court[:400].rstrip() + "…"

            self._derniere_note_lue = {"id": note["id"], "titre": note["titre"]}
            self._notifier(
                "archive_relue",
                f"📖 Avant de rêver, je relis « {note['titre']} »…",
            )
            _log.info(
                "BouclesArchives : relecture « %s » (id=%s)",
                note["titre"], note["id"],
            )
            return contenu_court
        except Exception as e:
            _log.warning("BouclesArchives.avant_reve a échoué : %s", e)
            return None

    def derniere_note_lue(self):
        """Retourne {'id', 'titre'} de la dernière note relue, ou None."""
        return dict(self._derniere_note_lue) if self._derniere_note_lue else None

    # ---------------------------------------------------------------- archivage

    def apres_reve(self, liste_reves, nom="Nyx"):
        """
        Archive automatiquement les rêves marquants.
        - Ne fait rien si la boucle est désactivée.
        - Ne fait rien si budget horaire dépassé.
        - Ne fait rien si les rêves ne sont pas "marquants".
        - Anti-doublon : ne réarchive pas un rêve déjà traité (par timestamp).
        Retourne la liste des notes créées (0, 1 ou plusieurs dicts).
        """
        if not getattr(self.cfg, "archives_reves_actif", True):
            return []
        if not liste_reves:
            return []

        with self._verrou:
            self._maj_heure()
            if self._archives_heure >= self.budget_heure:
                _log.info(
                    "BouclesArchives : budget horaire atteint (%d/%d), "
                    "archivage des rêves ignoré.",
                    self._archives_heure, self.budget_heure,
                )
                return []

        notes_creees = []
        for reve in liste_reves:
            try:
                # Anti-doublon
                if reve.timestamp in self._derniers_reves_archives:
                    continue

                # Filtre "marquant" : seulement les rêves qui ont du relief
                if not self._reve_marquant(reve):
                    _log.debug(
                        "BouclesArchives : rêve %d non marquant, pas d'archivage.",
                        reve.numero,
                    )
                    continue

                titre = f"🌙 Rêve {reve.numero}/2 — {reve.timestamp}"
                contenu = reve.texte

                resultat = self.archives.ecrire_note(
                    titre=titre,
                    contenu=contenu,
                    type_note="reve",
                    tags=["reve", "auto", f"T{reve.temperature:.1f}"],
                )
                notes_creees.append(resultat["note"])
                self._derniers_reves_archives.add(reve.timestamp)

                with self._verrou:
                    self._archives_heure += 1

                # Événement vers l'interface (toast + bulle)
                self._notifier(
                    "archive_reve",
                    f"🗜️ Rêve {reve.numero}/2 archivé : « {titre} »",
                )
                _log.info(
                    "BouclesArchives : rêve %d archivé (%d → %d octets, algo %s).",
                    reve.numero,
                    resultat["note"]["taille_originale"],
                    resultat["note"]["taille_compressee"],
                    resultat["note"]["algo"],
                )

                # Purge auto éventuelle (déjà gérée par Archives.ecrire_note)
                purge = resultat.get("purge") or {}
                if purge.get("purge") and purge.get("supprimees", 0) > 0:
                    self._notifier(
                        "archive_purge",
                        f"♻️ Purge automatique : "
                        f"{purge['supprimees']} note(s) supprimée(s).",
                    )

                # Budget : on arrête dès qu'on a atteint la limite
                with self._verrou:
                    if self._archives_heure >= self.budget_heure:
                        break
            except Exception as e:
                _log.warning(
                    "BouclesArchives : archivage du rêve %d échoué : %s",
                    getattr(reve, "numero", "?"), e,
                )
        return notes_creees

    # ---------------------------------------------------------------- interne

    @staticmethod
    def _reve_marquant(reve):
        """
        Un rêve est "marquant" si la charge émotionnelle a changé
        de manière significative pendant le rêve, OU si le texte
        contient des mots à forte charge.
        """
        try:
            v_avant, a_avant, _ = reve.humeur_avant
            v_apres, a_apres, _ = reve.humeur_apres
            delta_valence = abs(v_apres - v_avant)
            delta_activation = abs(a_apres - a_avant)
            if delta_valence >= 0.05 or delta_activation >= 0.10:
                return True
        except Exception:
            pass

        # Repli : analyse lexicale légère
        texte = (reve.texte or "").lower()
        mots_forts = (
            "peur", "angoisse", "cauchemar", "mort", "cri", "fuir",
            "joie", "lumière", "voler", "rire", "amour", "étoile",
            "brisé", "vide", "obscur", "souffle", "brûlant",
        )
        return any(m in texte for m in mots_forts)

    def _maj_heure(self):
        h = int(time.time() // 3600)
        if h != self._heure_courante:
            self._heure_courante = h
            self._archives_heure = 0

    def _notifier(self, genre, texte):
        if self.sur_evenement is None:
            return
        try:
            self.sur_evenement(genre, texte)
        except Exception as e:
            _log.debug("BouclesArchives : callback a échoué : %s", e)
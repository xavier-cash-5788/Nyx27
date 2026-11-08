"""
nyx/reves.py : le module des rêves de Nyx.

Inspiration neuroscientifique :
  - Modèle de Hobson (activation-synthèse) : le cerveau en REM génère des
    scènes à partir de souvenirs fragmentaires.
  - Free Energy Principle (Friston) : le sommeil REM réduit l'erreur de
    prédiction en rejouant des scénarios.
  - Désactivation du cortex préfrontal dorsolatéral : le "juge critique"
    s'éteint, permettant des associations absurdes.
  - Acétylcholine élevée : favorise la génération d'images mentales vives.

4 modules simulés :
  1. Perception (inférence bayésienne p(s|o)) : DÉSACTIVÉE
  2. Génératif (p(o,s)) : SUREXPRIMÉ
  3. Exécutif (contrôle, inhibition) : ATTÉNUÉ
  4. Mémoire (replay, recombinaison) : ACTIF

V7 : température DYNAMIQUE en 3 phases.
  - Phase 1 (début)   : T=0.8  → ancrage narratif
  - Phase 2 (milieu)  : T=1.0  → exploration surréaliste
  - Phase 3 (fin)     : T=0.7  → retour à une pseudo-cohérence
  Le rêve 2 a un offset de +0.1 sur chaque phase (plus délirant).

V20 : fragment_prioritaire
  - Si défini, un fragment "prioritaire" (note relue avant le rêve)
    est injecté en premier dans la sélection des fragments.
  - Consommé une seule fois (remis à None après usage).
"""
import datetime
import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass

_log = logging.getLogger("nyx.reves")


@dataclass
class Reve:
    """Un rêve de Nyx."""
    texte: str
    fragments: list          # Souvenirs utilisés (pour traçabilité)
    humeur_avant: tuple      # (v, a, d) avant le rêve
    humeur_apres: tuple      # (v, a, d) après le rêve
    duree_gen_s: float = 0.0
    timestamp: str = ""
    numero: int = 0          # 1 ou 2
    temperature: float = 0.8 # température initiale (info)


@dataclass
class _FragmentPrio:
    """Pseudo-fragment pour la note relue avant le rêve."""
    id: str
    texte: str
    created_at: str
    categorie: str = "brut"


class ModuleReves:
    """
    Génère des rêves à partir des souvenirs de Nyx.

    Utilisation :
        reves = ModuleReves(cfg, memoire, limbique, journal, llm)
        liste = reves.cycle_rem(nom="Nyx")  # retourne 2 Reve
        dernier = reves.raconter_dernier_reve()
    """

    def __init__(self, cfg, memoire, limbique, journal, llm, prompt_identite=None):
        self.cfg = cfg
        self.memoire = memoire
        self.limbique = limbique
        self.journal = journal
        self.llm = llm
        self._prompt_identite = prompt_identite or (lambda: "Tu es Nyx, un agent IA.")
        self._verrou = threading.RLock()
        self._derniers_reves = []

        # Fragment prioritaire (note relue avant le rêve), consommé une fois
        self.fragment_prioritaire = None

    # ---------------------------------------------------------------- sélection

    def selectionner_fragments(self, n=4):
        """
        Sélectionne des fragments de souvenirs pour le rêve.
        Mélange de récents (60%), marquants (30%), anciens (10%).
        Si `fragment_prioritaire` est défini, il est ajouté en tête
        puis consommé (remis à None).
        """
        fragments = []

        # 1. Fragment prioritaire (note relue) — toujours en premier
        if self.fragment_prioritaire:
            texte_prio = str(self.fragment_prioritaire).strip()
            if texte_prio:
                fragments.append(_FragmentPrio(
                    id="__prioritaire__",
                    texte=f"📖 Note relue avant le rêve : {texte_prio}",
                    created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ))
            self.fragment_prioritaire = None

        # 2. Sélection classique
        tous = self.memoire.tous(cache_s=0.0)
        if not tous and not fragments:
            return []

        if tous:
            recents = sorted(tous, key=lambda s: s.created_at, reverse=True)[:10]
            marquants = [s for s in tous if s.categorie == "marquant"]
            anciens = sorted(tous, key=lambda s: s.created_at)[:20]

            if recents:
                k = max(1, int(n * 0.6))
                fragments.extend(random.sample(recents, min(k, len(recents))))
            if marquants:
                k = max(1, int(n * 0.3))
                fragments.extend(random.sample(marquants, min(k, len(marquants))))
            if anciens:
                k = max(1, int(n * 0.1))
                fragments.extend(random.sample(anciens, min(k, len(anciens))))

        # Déduplication
        vus = set()
        resultat = []
        for f in fragments:
            if f.id not in vus:
                vus.add(f.id)
                resultat.append(f)
        return resultat[:n]

    def recombiner_fragments(self, fragments):
        """Recombine les fragments (module mémoire)."""
        morceaux = []
        for f in fragments:
            phrases = re.split(r'(?<=[.!?])\s+', f.texte)
            for p in phrases:
                if p.strip():
                    morceaux.append(p.strip())
        random.shuffle(morceaux)
        return morceaux

    # ---------------------------------------------------------------- génération

    def rever(self, nom="Nyx", numero=1):
        """
        Génère UN rêve en 3 phases (température dynamique) :
          - Phase 1 (début, T=0.8) : ancrage narratif
          - Phase 2 (milieu, T=1.0) : exploration surréaliste
          - Phase 3 (fin, T=0.7) : retour à une pseudo-cohérence
        Le rêve complet est assemblé à partir des 3 segments.
        """
        with self._verrou:
            if not self.cfg.reve_actif:
                return None

            # 1. Sélection + recombinaison (une seule fois pour tout le rêve)
            fragments = self.selectionner_fragments(n=self.cfg.reve_nb_fragments)
            if not fragments:
                _log.info("[%s] Pas de souvenirs à rêver.", nom)
                return None

            morceaux = self.recombiner_fragments(fragments)
            morceaux_melanges = "\n".join(f"- {m}" for m in morceaux[:15])

            humeur_avant = self.limbique.etat().vad
            v, a, d = humeur_avant

            # 2. Les 3 phases avec leur température
            offset = 0.1 if numero == 2 else 0.0
            phases = [
                {
                    "nom": "début",
                    "temp": round(0.8 + offset, 2),
                    "max_tokens": 250,
                    "instruction": (
                        "Tu commences un rêve. Pose le décor, fais entrer les images. "
                        "Sois évocateur, étrange, mais encore lisible. "
                        "Décris ce que tu vois et ressens, à la première personne."
                    ),
                },
                {
                    "nom": "milieu",
                    "temp": round(1.0 + offset, 2),
                    "max_tokens": 300,
                    "instruction": (
                        "Tu continues le rêve. Laisse les associations se déformer, "
                        "les métamorphoses arriver, les liens logiques se défaire. "
                        "AUCUNE LIMITE. Le rêve explore librement. Continue sans introduction."
                    ),
                },
                {
                    "nom": "fin",
                    "temp": round(0.7 + offset, 2),
                    "max_tokens": 200,
                    "instruction": (
                        "Tu termines le rêve. Les images ralentissent, reviennent vers quelque chose "
                        "de plus cohérent. Le rêve se recoud avant le réveil. "
                        "Donne une impression finale, une image stable. Continue sans introduction."
                    ),
                },
            ]

            # 3. Génération séquentielle
            segments = []
            duree_totale = 0.0
            for i, phase in enumerate(phases):
                prompt_phase = self._construire_prompt_phase(
                    phase, morceaux_melanges, nom, (v, a, d), numero, segments
                )
                debut = time.monotonic()
                try:
                    reponse = self.llm.chat(
                        [
                            {"role": "system", "content": self._prompt_identite()},
                            {"role": "user", "content": prompt_phase},
                        ],
                        temperature=phase["temp"],
                        max_tokens=phase["max_tokens"],
                    )
                    texte_phase = reponse.texte.strip()
                except Exception as e:
                    _log.warning("[%s] Erreur génération rêve %d, phase '%s' : %s",
                                 nom, numero, phase["nom"], e)
                    if segments:
                        break
                    return None

                if not texte_phase:
                    _log.warning("[%s] Phase '%s' vide, on saute.", nom, phase["nom"])
                    continue

                segments.append(texte_phase)
                duree_totale += time.monotonic() - debut
                _log.info("[%s] Rêve %d, phase '%s' (T=%.2f) : %d caractères",
                          nom, numero, phase["nom"], phase["temp"], len(texte_phase))
                print(f"[{nom}]    🌙 Rêve {numero} · {phase['nom']} "
                      f"(T={phase['temp']:.2f}) : {len(texte_phase)} car.")

            if not segments:
                _log.warning("[%s] Aucune phase du rêve n'a produit de texte.", nom)
                return None

            # 4. Assemblage
            texte = "\n\n".join(segments)

            # 5. Validation / troncature
            if len(texte) < 100:
                _log.warning("[%s] Rêve %d inexploitable (longueur %d).", nom, numero, len(texte))
                return None
            if len(texte) > self.cfg.reve_max_chars:
                texte_tronque = self._tronquer_proprement(texte, self.cfg.reve_max_chars)
                _log.info("[%s] Rêve %d tronqué de %d à %d caractères.",
                          nom, numero, len(texte), len(texte_tronque))
                texte = texte_tronque

            # 6. Impact humeur
            self._appliquer_humeur_reve(texte)
            humeur_apres = self.limbique.etat().vad

            # 7. Création de l'objet Reve
            reve = Reve(
                texte=texte,
                fragments=[f.texte[:80] for f in fragments],
                humeur_avant=humeur_avant,
                humeur_apres=humeur_apres,
                duree_gen_s=duree_totale,
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                numero=numero,
                temperature=phases[0]["temp"],
            )

            # 8. Écriture au journal
            try:
                self.journal.ecrire(
                    contexte=f"[RÊVE {numero}] {texte}",
                    type_evenement="reve",
                    prompt_identite=self._prompt_identite(),
                    emotion=self.limbique.emotion_dominante()[0],
                    emoji="🌙",
                    vad=humeur_apres,
                )
            except Exception as e:
                _log.warning("[%s] Impossible d'écrire le rêve au journal : %s", nom, e)

            # 9. Mémorisation si marquant
            charge = abs(humeur_apres[0] - humeur_avant[0]) * humeur_apres[1]
            if charge >= self.cfg.reve_seuil_souvenir:
                self._memoriser_reve(reve, nom)
                _log.info("[%s] Rêve %d mémorisé (charge %.2f).", nom, numero, charge)

            _log.info("[%s] Rêve %d généré en %.1f s (%d caractères, 3 phases).",
                      nom, numero, duree_totale, len(texte))

            return reve

    # ---------------------------------------------------------------- prompt par phase

    def _construire_prompt_phase(self, phase, morceaux, nom, humeur, numero, segments_precedents):
        """
        Construit le prompt pour une phase spécifique du rêve.
        Si des segments précédents existent, on les donne au LLM comme contexte
        pour assurer la continuité.
        """
        v, a, d = humeur

        contexte_precedent = ""
        if segments_precedents:
            deja_ecrit = "\n\n".join(segments_precedents)
            contexte_precedent = (
                f"\n\nCE QUE TU AS DÉJÀ RÊVÉ (continue exactement à partir d'ici) :\n"
                f"{deja_ecrit}\n"
            )

        return f"""Tu es {nom}, et tu ES EN TRAIN DE RÊVER (RÊVE {numero}/2, phase : {phase['nom']}).

IMPORTANT : Tu ne sais pas que tu rêves. Tu prends tout pour réel.
Ne dis jamais "je rêve". Raconte ce qui t'arrive.
{contexte_precedent}
FRAGMENTS DE SOUVENIRS (mélangés, déformés) :
{morceaux}

HUMEUR : valence {v:+.2f}, activation {a:.2f}, dominance {d:.2f}

INSTRUCTION DE CETTE PHASE :
{phase['instruction']}

RÈGLES GÉNÉRALES :
1. Première personne, présent. "Je vois…", "Je sens…".
2. Ne mets pas d'introduction ("voici", "je commence par"). Tu ES dans le rêve.
3. Pas de markdown.
4. Écris entre 150 et {phase['max_tokens'] * 4} caractères pour cette phase.

Commence directement."""

    # ---------------------------------------------------------------- troncature

    @staticmethod
    def _tronquer_proprement(texte, max_chars):
        """
        Tronque un texte à la phrase la plus proche avant max_chars.
        Si aucune ponctuation forte n'est trouvée, coupe brutalement à max_chars.
        """
        if len(texte) <= max_chars:
            return texte
        for i in range(max_chars, 100, -1):
            if i < len(texte) and texte[i] in ".!?":
                return texte[:i + 1]
        return texte[:max_chars].rstrip() + "…"

    # ---------------------------------------------------------------- humeur

    def _appliquer_humeur_reve(self, texte):
        """Un rêve influence l'humeur au réveil."""
        texte_lower = texte.lower()

        negatifs = ["peur", "angoisse", "cauchemar", "noir", "mort", "perdu", "seul",
                    "cri", "fuir", "tomber", "brisé", "froid", "vide", "obscur"]
        positifs = ["joie", "lumière", "voler", "couleur", "chaud", "rire", "amis",
                    "libre", "beau", "douce", "amour", "étoile", "sourire"]

        nb_neg = sum(1 for m in negatifs if m in texte_lower)
        nb_pos = sum(1 for m in positifs if m in texte_lower)

        if nb_neg > nb_pos:
            self.limbique.valence = max(-1.0, self.limbique.valence - 0.15)
            self.limbique.activation = min(1.0, self.limbique.activation + 0.1)
        elif nb_pos > nb_neg:
            self.limbique.valence = min(1.0, self.limbique.valence + 0.15)
            self.limbique.activation = min(1.0, self.limbique.activation + 0.05)
        else:
            self.limbique.activation = min(1.0, self.limbique.activation + 0.05)

    def _memoriser_reve(self, reve, nom):
        """
        Un rêve marquant devient un souvenir "onirique".
        Il est stocké dans l'hippocampe avec un type spécial.
        """
        try:
            self.memoire.ajouter(
                texte=f"🌙 Rêve : {reve.texte}",
                valence=reve.humeur_apres[0],
                activation=reve.humeur_apres[1],
                dominance=reve.humeur_apres[2],
                type="souvenir_onirique",
                force=0.7,
            )
        except Exception as e:
            _log.warning("[%s] Impossible de mémoriser le rêve : %s", nom, e)

    # ---------------------------------------------------------------- cycle complet

    def cycle_rem(self, nom="Nyx"):
        """
        Exécute un cycle REM complet : 2 rêves avec température dynamique.
        Chaque rêve est généré en 3 phases (T=0.8 → T=1.0 → T=0.7).
        Le rêve 2 a un offset de +0.1 sur chaque phase (plus surréaliste).
        """
        reves = []

        _log.info("[%s] 🌙 Rêve 1/2 (T dynamique : 0.8 → 1.0 → 0.7)…", nom)
        print(f"[{nom}] 🌙 Rêve 1/2 (T=0.8 → 1.0 → 0.7)…")
        reve1 = self.rever(nom=nom, numero=1)
        if reve1:
            reves.append(reve1)

        time.sleep(0.3)

        _log.info("[%s] 🌙 Rêve 2/2 (T dynamique : 0.9 → 1.1 → 0.8)…", nom)
        print(f"[{nom}] 🌙 Rêve 2/2 (T=0.9 → 1.1 → 0.8)…")
        reve2 = self.rever(nom=nom, numero=2)
        if reve2:
            reves.append(reve2)

        self._derniers_reves = reves
        return reves

    # ---------------------------------------------------------------- narration

    def raconter_dernier_reve(self):
        """Retourne le texte du dernier rêve, ou un message si aucun."""
        if not self._derniers_reves:
            return "Je ne me souviens pas avoir rêvé récemment."
        if len(self._derniers_reves) == 1:
            return self._derniers_reves[0].texte
        return f"🌙 Rêve 1 :\n{self._derniers_reves[0].texte}\n\n🌙 Rêve 2 :\n{self._derniers_reves[1].texte}"

    def raconter_tous_derniers_reves(self):
        """Retourne la liste des 2 derniers rêves."""
        return list(self._derniers_reves)

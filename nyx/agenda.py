"""
nyx/agenda.py : l'agenda de Nyx (événements datés + rappels).

Stockage JSON, écriture atomique. Pas de récurrences en V1.
Le module est PUR : il ne connaît ni l'horloge interne ni le LLM. C'est l'appelant
(Pulsions.tick, agent, interface) qui décide quand interroger les rappels.

Compatible avec deux types de "maintenant" :
  - datetime  (utilisé par l'interface et les tests)
  - float     (timestamp epoch, utilisé par Pulsions.tick via time.time())

La fonction interne `_normaliser_datetime()` convertit les deux en datetime,
ce qui évite le bug "unsupported operand type(s) for -: 'datetime.datetime' and 'float'".
"""
import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

FORMAT_DATE = "%Y-%m-%d"
FORMAT_HEURE = "%H:%M"
FORMAT_DATETIME = "%Y-%m-%d %H:%M"
FICHIER_DEFAUT = "agenda.json"


def _normaliser_datetime(valeur):
    """
    Convertit une valeur temporelle hétérogène en datetime.

    Accepte :
      - None       → datetime.now()
      - datetime   → retourné tel quel
      - int / float → epoch UNIX, converti via fromtimestamp()
      - autre      → datetime.now() (repli sûr)
    """
    if valeur is None:
        return datetime.now()
    if isinstance(valeur, datetime):
        return valeur
    if isinstance(valeur, (int, float)):
        try:
            return datetime.fromtimestamp(valeur)
        except (OSError, ValueError, OverflowError):
            return datetime.now()
    return datetime.now()


@dataclass
class Evenement:
    id: str
    titre: str
    date: str
    heure: str
    duree_min: int
    description: str
    rappel_min: int
    cree_le: str
    rappele_le: str = ""

    @property
    def datetime_debut(self):
        """datetime de début de l'événement (avec heure si précisée, sinon minuit)."""
        if self.heure:
            return datetime.strptime(f"{self.date} {self.heure}", FORMAT_DATETIME)
        return datetime.strptime(self.date, FORMAT_DATE)

    def minutes_avant(self, maintenant=None):
        """
        Minutes avant le début de l'événement (négatif si déjà passé).
        Accepte un datetime OU un timestamp epoch (float) pour `maintenant`.
        """
        maintenant = _normaliser_datetime(maintenant)
        delta = self.datetime_debut - maintenant
        return delta.total_seconds() / 60

    def en_dict(self):
        return asdict(self)

    def resumer(self):
        quand = self.date
        if self.heure:
            quand = f"{self.date} à {self.heure}"
        suffixe = f" — {self.description}" if self.description else ""
        return f"[{self.id}] {self.titre} — {quand}{suffixe}"


class Agenda:
    def __init__(self, fichier=FICHIER_DEFAUT):
        self.fichier = Path(fichier)
        self._evenements = {}
        self._charger()

    # ------------------------------------------------------------------ I/O
    def _charger(self):
        if not self.fichier.is_file():
            return
        try:
            data = json.loads(self.fichier.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for brut in data.get("evenements", []):
            try:
                ev = Evenement(**brut)
                self._evenements[ev.id] = ev
            except (TypeError, KeyError):
                continue

    def _sauver(self):
        data = {"version": 1, "evenements": [e.en_dict() for e in self._evenements.values()]}
        tmp = self.fichier.with_suffix(self.fichier.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.fichier)

    # ------------------------------------------------------------------ CRUD
    def ajouter(self, titre, date=None, heure="", duree_min=60, description="", rappel_min=15):
        if not titre or not titre.strip():
            raise ValueError("titre vide")
        date = (date or datetime.now().strftime(FORMAT_DATE)).strip()
        try:
            datetime.strptime(date, FORMAT_DATE)
        except ValueError:
            raise ValueError(f"date invalide : {date!r} (attendu YYYY-MM-DD)")
        if heure:
            try:
                datetime.strptime(heure, FORMAT_HEURE)
            except ValueError:
                raise ValueError(f"heure invalide : {heure!r} (attendu HH:MM)")
        ev = Evenement(
            id=str(uuid.uuid4())[:8],
            titre=titre.strip(),
            date=date,
            heure=heure,
            duree_min=max(0, int(duree_min)),
            description=description.strip(),
            rappel_min=max(0, int(rappel_min)),
            cree_le=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._evenements[ev.id] = ev
        self._sauver()
        return ev.id

    def supprimer(self, id_ev):
        if id_ev in self._evenements:
            del self._evenements[id_ev]
            self._sauver()
            return True
        return False

    def modifier(self, id_ev, **champs):
        ev = self._evenements.get(id_ev)
        if not ev:
            return False
        for cle, val in champs.items():
            if hasattr(ev, cle) and cle != "id":
                setattr(ev, cle, val)
        self._sauver()
        return True

    # ------------------------------------------------------------------ requêtes
    def tous(self):
        return sorted(self._evenements.values(), key=lambda e: e.datetime_debut)

    def du_jour(self, jour=None):
        jour = jour or datetime.now().strftime(FORMAT_DATE)
        return [e for e in self.tous() if e.date == jour]

    def de_la_semaine(self, depuis=None):
        """
        Événements des 7 prochains jours CALENDAIRES (aujourd'hui inclus, même si
        l'heure est déjà passée).
        """
        depuis = _normaliser_datetime(depuis)
        debut_jour = depuis.replace(hour=0, minute=0, second=0, microsecond=0)
        fin = debut_jour + timedelta(days=7)
        return [e for e in self.tous() if debut_jour <= e.datetime_debut < fin]

    def a_venir(self, n=10, maintenant=None):
        """Les n prochains événements à partir de `maintenant`."""
        maintenant = _normaliser_datetime(maintenant)
        return [e for e in self.tous() if e.datetime_debut >= maintenant][:n]

    def a_rappeler(self, maintenant=None, tolerance_min=1):
        """
        Événements dont le rappel doit être émis MAINTENANT.

        Un événement est rappelé si :
          - il n'a pas déjà été rappelé (rappele_le vide) ;
          - son début est dans moins de `rappel_min` minutes (ou déjà passé de moins
            de `tolerance_min` minutes, pour ne pas rater un rappel à cause d'un tick).

        `maintenant` peut être un datetime OU un timestamp epoch (float).
        """
        maintenant = _normaliser_datetime(maintenant)
        resultat = []
        for ev in self.tous():
            if ev.rappele_le:
                continue
            minutes = ev.minutes_avant(maintenant)
            if minutes < -tolerance_min:
                continue
            if minutes <= ev.rappel_min:
                resultat.append(ev)
        return resultat

    def marquer_rappele(self, id_ev):
        ev = self._evenements.get(id_ev)
        if ev and not ev.rappele_le:
            ev.rappele_le = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._sauver()

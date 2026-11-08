"""
nyx/archives.py : les archives compressées de Nyx.

Ce module permet à Nyx de stocker des notes dans un format compressé
(.cpk), de les relire à la volée, et de gérer automatiquement un plafond
de taille (1 Go par défaut).

Architecture :
  - Chaque note devient un fichier .cpk dans archives/
  - Un index JSON (archives/index.json) liste les métadonnées
  - Compression "media-smart" : adaptée au type de contenu
  - Plafond global : purge automatique des plus anciennes quand on dépasse

Format .cpk (rappel) :
  [4o magic][1o algo][8o taille originale][4o taille nom][nom UTF-8][données]

API publique :
  archives.ecrire_note(titre, contenu, type_note="note") -> dict
  archives.lire_note(id_note) -> str ou None
  archives.lister(filtre=None) -> List[dict]
  archives.supprimer(id_note) -> bool
  archives.restaurer(id_note) -> str ou None
  archives.stats() -> dict
  archives.purger_jusqua_limite() -> dict
"""
import bz2
import json
import lzma
import logging
import os
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

_log = logging.getLogger("nyx.archives")

# ==================================================================== format .cpk

MAGIC = b"CPK\x02"
ALGO_ID = {"store": 0, "zlib": 1, "bz2": 2, "lzma": 3, "lzma_delta": 4}
ID_ALGO = {v: k for k, v in ALGO_ID.items()}
EXT_OUT = ".cpk"

# ==================================================================== classification

ALREADY_COMPRESSED = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".avif",
    ".mp4", ".mkv", ".avi", ".mov", ".mp3", ".aac", ".ogg", ".flac",
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".pdf", ".epub",
}

UNCOMPRESSED_MEDIA = {".wav", ".bmp", ".tif", ".tiff", ".raw"}

TEXT_LIKE = {
    ".txt", ".md", ".json", ".csv", ".xml", ".html", ".css", ".js",
    ".py", ".c", ".cpp", ".h", ".java", ".rs", ".go", ".sh", ".log",
}


# ==================================================================== delta + compression

def delta_encode(data: bytes) -> bytes:
    if not data:
        return data
    out = bytearray(len(data))
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = (data[i] - data[i - 1]) & 0xFF
    return bytes(out)


def delta_decode(data: bytes) -> bytes:
    if not data:
        return data
    out = bytearray(len(data))
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = (data[i] + out[i - 1]) & 0xFF
    return bytes(out)


def choose_algo(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in ALREADY_COMPRESSED:
        return "store"
    if ext in UNCOMPRESSED_MEDIA:
        return "lzma_delta"
    if ext in TEXT_LIKE:
        return "lzma"
    return "auto"


def _compress(data: bytes, algo: str, level: int) -> bytes:
    if algo == "store":
        return data
    if algo == "zlib":
        return zlib.compress(data, level)
    if algo == "bz2":
        return bz2.compress(data, level)
    if algo == "lzma":
        return lzma.compress(data, preset=min(level, 9))
    if algo == "lzma_delta":
        return lzma.compress(delta_encode(data), preset=min(level, 9))
    raise ValueError(f"algo inconnu : {algo}")


def _decompress(data: bytes, algo: str) -> bytes:
    if algo == "store":
        return data
    if algo == "zlib":
        return zlib.decompress(data)
    if algo == "bz2":
        return bz2.decompress(data)
    if algo == "lzma":
        return lzma.decompress(data)
    if algo == "lzma_delta":
        return delta_decode(lzma.decompress(data))
    raise ValueError(f"algo inconnu : {algo}")


def _best(data: bytes, level: int):
    best_algo, best_data = "store", data
    for algo in ("zlib", "bz2", "lzma"):
        try:
            c = _compress(data, algo, level)
            if len(c) < len(best_data):
                best_algo, best_data = algo, c
        except Exception:
            pass
    return best_algo, best_data


def pack_bytes(data: bytes, filename: str, level: int = 9) -> tuple:
    """
    Compresse `data` en .cpk. Retourne (données_cpk, algo, taille_originale).
    """
    algo = choose_algo(filename)
    if algo == "auto":
        algo, compressed = _best(data, level)
    else:
        compressed = _compress(data, algo, level)
    if len(compressed) >= len(data) and algo != "store":
        algo, compressed = "store", data
    name_b = os.path.basename(filename).encode("utf-8")
    header = MAGIC + bytes([ALGO_ID[algo]]) + struct.pack(">QI", len(data), len(name_b)) + name_b
    return header + compressed, algo, len(data)


def unpack_bytes(raw: bytes) -> tuple:
    """
    Décompresse un .cpk. Retourne (données_originales, filename, algo, taille_originale).
    """
    if len(raw) < 17 or raw[:4] != MAGIC:
        raise ValueError("Fichier .cpk invalide ou corrompu")
    algo_id = raw[4]
    if algo_id not in ID_ALGO:
        raise ValueError(f"algo_id inconnu : {algo_id}")
    algo = ID_ALGO[algo_id]
    orig, name_len = struct.unpack(">QI", raw[5:17])
    name = raw[17:17 + name_len].decode("utf-8")
    payload = raw[17 + name_len:]
    data = _decompress(payload, algo)
    if len(data) != orig:
        raise ValueError("Taille incorrecte : fichier corrompu ?")
    return data, name, algo, orig


# ==================================================================== dataclass note

@dataclass
class NoteArchivee:
    """Une note archivée."""
    id: str
    titre: str
    type_note: str  # "note", "journal", "document", "souvenir", "custom"
    fichier: str  # nom du .cpk dans archives/
    taille_originale: int
    taille_compressee: int
    algo: str
    date_creation: str
    tags: list = field(default_factory=list)
    preview: str = ""  # 200 premiers caractères pour l'interface

    def en_dict(self):
        return asdict(self)


# ==================================================================== Archives

class Archives:
    """
    Gestionnaire d'archives compressées pour Nyx.

    Utilisation :
        archives = Archives(cfg)
        archives.ecrire_note("Ma première note", "Contenu de la note...")
        archives.lister()
        archives.lire_note(id_note)
    """

    def __init__(self, cfg, dossier=None):
        self.cfg = cfg
        self.dossier = Path(dossier or getattr(cfg, "archives_dossier", "archives"))
        self.dossier.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dossier / "index.json"
        self._verrou = threading.RLock()
        self._index = self._charger_index()

    # ---------------------------------------------------------------- index

    def _charger_index(self) -> dict:
        if not self.index_path.is_file():
            return {"notes": [], "version": 1}
        try:
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "notes" not in data:
                return {"notes": [], "version": 1}
            return data
        except (OSError, ValueError) as e:
            _log.warning("Index archives illisible (%s) : recréé", e)
            return {"notes": [], "version": 1}

    def _sauver_index(self):
        tmp = str(self.index_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index_path)

    # ---------------------------------------------------------------- taille

    def _taille_totale(self) -> int:
        """Taille totale des .cpk sur disque (en octets)."""
        total = 0
        for n in self._index["notes"]:
            chemin = self.dossier / n["fichier"]
            if chemin.is_file():
                total += chemin.stat().st_size
        return total

    def _taille_limite(self) -> int:
        return int(getattr(self.cfg, "archives_limite_go", 1.0) * (1024 ** 3))

    # ---------------------------------------------------------------- écriture

    def ecrire_note(self, titre: str, contenu: str, type_note: str = "note",
                    tags: list = None) -> dict:
        """
        Compresse et archive une note.

        Retourne un dict avec les métadonnées de la note archivée.
        """
        with self._verrou:
            if not titre or not contenu:
                raise ValueError("titre et contenu obligatoires")

            # 1. Compresse
            nom_fichier = f"{_slug(titre)}-{uuid.uuid4().hex[:8]}.txt"
            data = contenu.encode("utf-8")
            try:
                cpk_bytes, algo, orig = pack_bytes(data, nom_fichier)
            except Exception as e:
                raise RuntimeError(f"Compression impossible : {e}") from e

            # 2. Écrit le .cpk
            cpk_path = self.dossier / (nom_fichier + EXT_OUT)
            with open(cpk_path, "wb") as f:
                f.write(cpk_bytes)

            # 3. Crée la métadonnée
            note = NoteArchivee(
                id=str(uuid.uuid4()),
                titre=titre[:200],
                type_note=type_note,
                fichier=cpk_path.name,
                taille_originale=orig,
                taille_compressee=len(cpk_bytes),
                algo=algo,
                date_creation=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tags=tags or [],
                preview=contenu[:200],
            )

            # 4. Ajoute à l'index
            self._index["notes"].append(note.en_dict())
            self._sauver_index()

            # 5. Purge automatique si nécessaire
            purge = self.purger_jusqua_limite()

            _log.info("🗜️ Note archivée : « %s » (%d → %d octets, algo %s)",
                      titre[:50], orig, len(cpk_bytes), algo)
            return {
                "note": note.en_dict(),
                "purge": purge,
            }

    # ---------------------------------------------------------------- lecture

    def lire_note(self, id_note: str) -> str:
        """Lit une note archivée. Retourne le contenu ou None si absente."""
        with self._verrou:
            meta = next((n for n in self._index["notes"] if n["id"] == id_note), None)
            if meta is None:
                return None
            chemin = self.dossier / meta["fichier"]
            if not chemin.is_file():
                _log.warning("Fichier .cpk introuvable : %s", chemin)
                return None
            try:
                with open(chemin, "rb") as f:
                    data, _name, _algo, _orig = unpack_bytes(f.read())
                return data.decode("utf-8")
            except Exception as e:
                _log.warning("Erreur décompression %s : %s", chemin.name, e)
                return None

    # ---------------------------------------------------------------- listing

    def lister(self, filtre: str = None, tri: str = "date_desc") -> list:
        """
        Liste les notes archivées.
        filtre : None, "note", "journal", "document", etc.
        tri    : "date_desc", "date_asc", "taille_desc", "taille_asc"
        """
        with self._verrou:
            notes = list(self._index["notes"])
            if filtre:
                notes = [n for n in notes if n.get("type_note") == filtre]
            if tri == "date_desc":
                notes.sort(key=lambda n: n.get("date_creation", ""), reverse=True)
            elif tri == "date_asc":
                notes.sort(key=lambda n: n.get("date_creation", ""))
            elif tri == "taille_desc":
                notes.sort(key=lambda n: n.get("taille_originale", 0), reverse=True)
            elif tri == "taille_asc":
                notes.sort(key=lambda n: n.get("taille_originale", 0))
            return notes

    def chercher(self, mot_cle: str) -> list:
        """Cherche dans les titres et previews."""
        if not mot_cle:
            return []
        m = mot_cle.lower()
        return [n for n in self._index["notes"]
                if m in n.get("titre", "").lower() or m in n.get("preview", "").lower()]

    # ---------------------------------------------------------------- suppression

    def supprimer(self, id_note: str) -> bool:
        """Supprime une note archivée (fichier + index)."""
        with self._verrou:
            for i, n in enumerate(self._index["notes"]):
                if n["id"] == id_note:
                    chemin = self.dossier / n["fichier"]
                    try:
                        if chemin.is_file():
                            chemin.unlink()
                    except OSError as e:
                        _log.warning("Suppression fichier impossible : %s", e)
                    del self._index["notes"][i]
                    self._sauver_index()
                    return True
            return False

    # ---------------------------------------------------------------- purge

    def purger_jusqua_limite(self) -> dict:
        """
        Purge les plus anciennes notes jusqu'à redescendre sous le plafond.
        Retourne un dict avec le bilan.
        """
        with self._verrou:
            limite = self._taille_limite()
            total = self._taille_totale()
            if total <= limite:
                return {"purge": False, "supprimees": 0}

            # Trie par date (plus anciennes d'abord)
            notes = sorted(self._index["notes"], key=lambda n: n.get("date_creation", ""))
            supprimees = 0
            for n in notes:
                if total <= limite * 0.9:  # marge de 10 %
                    break
                chemin = self.dossier / n["fichier"]
                try:
                    if chemin.is_file():
                        taille = chemin.stat().st_size
                        chemin.unlink()
                        total -= taille
                        supprimees += 1
                except OSError as e:
                    _log.warning("Purge impossible pour %s : %s", chemin.name, e)
                    continue
                self._index["notes"].remove(n)

            if supprimees:
                self._sauver_index()
                _log.warning("🗜️ Purge archives : %d note(s) supprimée(s), total %d → %d octets",
                            supprimees, total + sum(0 for _ in range(0)), total)

            return {
                "purge": True,
                "supprimees": supprimees,
                "total_apres": total,
                "limite": limite,
            }

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict:
        """Retourne un résumé de l'état des archives."""
        with self._verrou:
            total = self._taille_totale()
            limite = self._taille_limite()
            par_type = {}
            for n in self._index["notes"]:
                t = n.get("type_note", "note")
                par_type[t] = par_type.get(t, 0) + 1
            return {
                "total_notes": len(self._index["notes"]),
                "total_octets": total,
                "total_go": total / (1024 ** 3),
                "limite_go": limite / (1024 ** 3),
                "pourcentage": 100 * total / limite if limite > 0 else 0,
                "par_type": par_type,
                "dossier": str(self.dossier.absolute()),
            }

    # ---------------------------------------------------------------- export

    def exporter_tout(self, dossier_sortie: str) -> dict:
        """Exporte toutes les notes en clair (décompressées) dans un dossier."""
        with self._verrou:
            dest = Path(dossier_sortie)
            dest.mkdir(parents=True, exist_ok=True)
            exportees = 0
            for n in self._index["notes"]:
                contenu = self.lire_note(n["id"])
                if contenu is None:
                    continue
                nom_fichier = f"{_slug(n['titre'])}-{n['id'][:8]}.txt"
                try:
                    (dest / nom_fichier).write_text(contenu, encoding="utf-8")
                    exportees += 1
                except OSError as e:
                    _log.warning("Export %s impossible : %s", nom_fichier, e)
            return {"exportees": exportees, "dossier": str(dest.absolute())}


# ==================================================================== utilitaires

def _slug(texte: str, longueur_max: int = 60) -> str:
    """Transforme un titre en slug pour nom de fichier."""
    import re
    import unicodedata
    if not texte:
        return "note"
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
    texte = re.sub(r"[^a-zA-Z0-9_-]+", "_", texte).strip("_").lower()
    return texte[:longueur_max] or "note"


def _fmt_taille(n: int) -> str:
    for u in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{int(n) if u == 'o' else f'{n:.1f}'} {u}"
        n /= 1024
    return f"{n:.1f} To"
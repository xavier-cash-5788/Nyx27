"""
nyx/documents.py : documents texte nommés que Nyx peut créer et relire.

Un document est un simple fichier .txt dans un dossier (`documents/` par défaut).
Le nom est normalisé (slug) : "Notes de philo" → "notes_de_philo.txt".
Écriture atomique, pas de versionnage : on écrase ou on refuse.
"""
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DOSSIER_DEFAUT = "documents"
EXTENSION = ".txt"
LONGUEUR_NOM_MAX = 60


def normaliser_nom(nom):
    """Transforme un nom lisible en nom de fichier sûr (sans accent, sans espace)."""
    if not nom:
        return ""
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode("ascii")
    nom = re.sub(r"[^a-zA-Z0-9_-]+", "_", nom).strip("_").lower()
    return nom[:LONGUEUR_NOM_MAX]


@dataclass(frozen=True)
class InfoDocument:
    nom: str
    chemin: str
    taille: int
    modifie_le: str


class DocumentStore:
    def __init__(self, dossier=DOSSIER_DEFAUT):
        self.dossier = Path(dossier)
        self.dossier.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ lecture
    def _chemin(self, nom):
        n = normaliser_nom(nom)
        if not n:
            raise ValueError("nom de document vide")
        return self.dossier / f"{n}{EXTENSION}"

    def existe(self, nom):
        try:
            return self._chemin(nom).is_file()
        except ValueError:
            return False

    def lire(self, nom):
        chemin = self._chemin(nom)
        if not chemin.is_file():
            return None
        return chemin.read_text(encoding="utf-8")

    def lister(self):
        fichiers = sorted(self.dossier.glob(f"*{EXTENSION}"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        return [self._info(p) for p in fichiers]

    @staticmethod
    def _info(p):
        st = p.stat()
        return InfoDocument(
            nom=p.stem,
            chemin=str(p),
            taille=st.st_size,
            modifie_le=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )

    def chercher(self, requete, limite=5):
        """Cherche dans les noms ET les contenus. Retourne les documents les plus pertinents."""
        if not requete:
            return []
        mots = [m.lower() for m in re.findall(r"\w+", requete) if len(m) > 2]
        if not mots:
            return []
        scores = []
        for p in self.dossier.glob(f"*{EXTENSION}"):
            try:
                contenu = p.read_text(encoding="utf-8").lower()
            except OSError:
                continue
            nom = p.stem.lower()
            score = sum(contenu.count(m) + 3 * nom.count(m) for m in mots)
            if score > 0:
                scores.append((score, p))
        scores.sort(key=lambda x: -x[0])
        return [self._info(p) for _, p in scores[:limite]]

    # ------------------------------------------------------------------ écriture
    def creer(self, nom, contenu, ecraser=False):
        if not isinstance(contenu, str):
            raise ValueError("le contenu doit être une chaîne")
        chemin = self._chemin(nom)
        if chemin.exists() and not ecraser:
            raise FileExistsError(f"le document « {chemin.stem} » existe déjà")
        tmp = chemin.with_suffix(chemin.suffix + ".tmp")
        tmp.write_text(contenu, encoding="utf-8")
        os.replace(tmp, chemin)
        return str(chemin)

    def ajouter(self, nom, contenu):
        ancien = self.lire(nom) or ""
        separateur = "\n\n" if ancien and not ancien.endswith("\n\n") else ""
        return self.creer(nom, ancien + separateur + contenu, ecraser=True)

    def supprimer(self, nom):
        chemin = self._chemin(nom)
        if not chemin.is_file():
            return False
        chemin.unlink()
        return True
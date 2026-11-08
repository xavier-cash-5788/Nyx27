"""
nyx/intentions.py : détecte dans un message utilisateur les intentions.

Intentions reconnues :
  - "document"        : écrire un document (déclencheurs « écris-moi », etc.)
  - "agenda"          : ajouter un événement (déclencheurs « rappelle-moi », etc.)
  - "archive"         : archiver la dernière réponse (déclencheurs « Archive ça », etc.)
  - "lire_document"   : lire un document que Nyx a écrit (déclencheurs « lis ton document », etc.)
  - "lister_documents": lister les documents écrits (déclencheurs « quels documents », etc.)

Stratégie : mots-clés (rapide, pas de LLM) puis extraction structurée par le LLM
(une seule passe, JSON strict). Si le LLM échoue, on retombe sur une extraction
minimale basée sur le message brut.

Ordre de priorité dans detecter() :
  1. DECLENCHEURS_LIRE_DOC   (le plus spécifique : "lis ton document sur...")
  2. DECLENCHEURS_LISTER_DOC (très spécifique aussi)
  3. DECLENCHEURS_ARCHIVE    (avant doc/agenda : "sauvegarde ça", "note dans mes archives")
  4. DECLENCHEURS_DOC        ("écris-moi...")
  5. DECLENCHEURS_AGENDA     ("rappelle-moi...")
"""
from dataclasses import dataclass

DECLENCHEURS_DOC = (
    "écris-moi", "écris moi", "rédige-moi", "rédige moi", "rédige",
    "écris un document", "écris une note", "écris un texte",
    "note sur", "document sur", "texte sur",
    "fais-moi un texte", "fais moi un texte",
    "peux-tu écrire", "peux tu écrire", "peux-tu rédiger", "peux tu rédiger",
    "je veux un document", "crée un document", "créer un document",
)

DECLENCHEURS_AGENDA = (
    "rappelle-moi", "rappelle moi",
    "note dans mon agenda", "note dans l'agenda",
    "ajoute à mon agenda", "ajoute dans mon agenda",
    "ajoute un rendez-vous", "ajoute un rdv",
    "planifie", "programme un rendez-vous", "programme un rdv",
)

DECLENCHEURS_ARCHIVE = (
    "archive ça", "archive ceci", "archive cette", "archive ce",
    "note dans mes archives", "note dans les archives",
    "ajoute à mes archives", "ajoute dans les archives",
    "sauvegarde ça", "sauvegarde ceci",
)

DECLENCHEURS_LIRE_DOC = (
    "lis ton document", "lis le document", "lis-moi ton document",
    "lis moi ton document", "relis ton document", "relis le document",
    "qu'as-tu écrit sur", "qu as tu ecrit sur", "qu'as tu écrit sur",
    "montre-moi ton document", "montre moi ton document",
    "ouvre ton document", "ouvre le document",
    "lis ce que tu as écrit", "relis ce que tu as écrit",
)

DECLENCHEURS_LISTER_DOC = (
    "quels documents", "quels sont tes documents", "liste tes documents",
    "liste les documents", "liste-moi tes documents",
    "montre-moi tes documents", "montre moi tes documents",
    "tu as écrit quels documents", "qu'as-tu écrit", "qu as tu ecrit",
    "tes documents", "tous tes documents",
)


@dataclass
class Intention:
    type: str               # 'document' | 'agenda' | 'archive' | 'lire_document' | 'lister_documents'
    sujet: str = ""
    nom_propose: str = ""
    titre: str = ""
    date: str = ""
    heure: str = ""


class DetecteurIntention:
    def __init__(self, llm=None):
        self.llm = llm

    def detecter(self, message):
        if not message:
            return None
        bas = message.lower()

        # 1. Lecture de document (le plus spécifique : "lis ton document sur X")
        if any(d in bas for d in DECLENCHEURS_LIRE_DOC):
            return self._extraire_lire_document(message)

        # 2. Listing de documents
        if any(d in bas for d in DECLENCHEURS_LISTER_DOC):
            return Intention(type="lister_documents", sujet=message)

        # 3. Archives (évite que "sauvegarde ça" soit capté par "document")
        if any(d in bas for d in DECLENCHEURS_ARCHIVE):
            return Intention(type="archive", sujet=message)

        # 4. Écriture de document
        if any(d in bas for d in DECLENCHEURS_DOC):
            return self._extraire_document(message)

        # 5. Agenda
        if any(d in bas for d in DECLENCHEURS_AGENDA):
            return self._extraire_agenda(message)

        return None

    # ------------------------------------------------------------------ extraction
    def _extraire_document(self, message):
        sujet, nom = message, "document"
        if self.llm is not None:
            prompt = f"""Message de l'utilisateur :
<message>
{message}
</message>
L'utilisateur demande un document. Extrais :
- "sujet" : le sujet précis du document (1 phrase courte, 10-20 mots)
- "nom" : un nom de fichier court (max 40 car.), minuscules, sans accent, underscores
Réponds uniquement en JSON : {{"sujet": "...", "nom": "..."}}"""
            data = self._appel_json(prompt)
            if data:
                sujet = data.get("sujet") or sujet
                nom = data.get("nom") or nom
        from .documents import normaliser_nom
        return Intention(type="document", sujet=sujet, nom_propose=normaliser_nom(nom) or "document")

    def _extraire_agenda(self, message):
        from datetime import datetime
        aujourdhui = datetime.now().strftime("%Y-%m-%d")
        titre, date, heure = message, aujourdhui, ""
        if self.llm is not None:
            prompt = f"""Message de l'utilisateur :
<message>
{message}
</message>
L'utilisateur veut ajouter un événement à son agenda. Extrais :
- "titre" : le titre de l'événement (court, 3-8 mots)
- "date" : la date au format YYYY-MM-DD (aujourd'hui = {aujourdhui} ; "demain" = +1 jour)
- "heure" : l'heure au format HH:MM, ou "" si non précisée
Réponds uniquement en JSON : {{"titre": "...", "date": "YYYY-MM-DD", "heure": "HH:MM"}}"""
            data = self._appel_json(prompt)
            if data:
                titre = data.get("titre") or titre
                date = data.get("date") or date
                heure = data.get("heure") or ""
        return Intention(type="agenda", titre=titre, date=date, heure=heure)

    def _extraire_lire_document(self, message):
        """Identifie QUEL document lire (nom ou sujet)."""
        nom, sujet = "", message
        if self.llm is not None:
            # On liste les documents existants pour aider le LLM à choisir
            try:
                from .documents import DocumentStore
                docs = DocumentStore().lister()
                liste_docs = "\n".join(f"- {d.nom}" for d in docs[:30]) or "(aucun)"
            except Exception:
                liste_docs = "(liste indisponible)"
            prompt = f"""Message de l'utilisateur :
<message>
{message}
</message>
L'utilisateur demande à Nyx de lire un de ses documents.
Voici les documents existants :
<documents>
{liste_docs}
</documents>
Détermine quel document l'utilisateur veut lire.
- "nom" : le nom exact du document à lire (parmi la liste ci-dessus), ou "" si aucun ne correspond
- "sujet" : le sujet demandé par l'utilisateur (pour recherche si le nom est vide)
Réponds uniquement en JSON : {{"nom": "...", "sujet": "..."}}"""
            data = self._appel_json(prompt)
            if data:
                nom = data.get("nom") or ""
                sujet = data.get("sujet") or sujet
        from .documents import normaliser_nom
        return Intention(
            type="lire_document",
            sujet=sujet,
            nom_propose=normaliser_nom(nom) if nom else "",
        )

    def _appel_json(self, prompt):
        try:
            reponse = self.llm.chat([{"role": "user", "content": prompt}],
                                    format="json", temperature=0)
            if isinstance(reponse, dict):
                contenu = reponse.get("message", {}).get("content", "")
            else:
                contenu = getattr(reponse, "texte", "") or ""
            from .llm import extraire_json
            return extraire_json(contenu)
        except Exception:
            return None

"""
nyx/journal.py : le journal intime de Nyx (module 5/9).

Remplace, dans l'ancien main.py : _ecrire_dans_journal, _enregistrer_journal,
_epoch_de, _entrees_journal_triees, _dernieres_reflexions, _charger_identite_recente,
_catchup_identitaire.

Ce que l'ancien code faisait dans l'agent :
- écrivait dans le journal via un appel LLM avec un prompt dédié
- détectait les doublons par similarité Chroma (SEUIL_DOUBLON_JOURNAL)
- rafraîchissait l'identité récente à chaque entrée
- triait les entrées par epoch (avec repli sur timestamp pour l'ancien format)

Ce module ne connaît ni l'état émotionnel ni la mémoire long terme :
l'émotion et les dernières réflexions lui sont passées en paramètre.
Il reçoit une collection Chroma déjà ouverte (testable avec un faux Chroma).

NOUVEAU : si une `embedding_function` est fournie (typiquement OllamaEmbeddingFunction
pour nomic-embed-text), les vecteurs sont PRÉCALCULÉS ici, avec les bons préfixes Nomic :
  - "search_document: " pour l'ajout d'une entrée
  - "search_query: "    pour la détection de doublon (query)
Cela garantit la cohérence dimensionnelle avec la mémoire et évite que ChromaDB
recalcule ses propres embeddings (par défaut 384 dims, incompatibles avec nomic 768).
"""
import logging
import time
import uuid
from dataclasses import dataclass

_log = logging.getLogger("nyx.journal")

FORMAT_TS = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class EntreeJournal:
    """Une entrée du journal, lue en triant par epoch."""
    epoch: float
    texte: str
    type_evenement: str
    emotion: str
    timestamp: str
    valence: float
    activation: float
    dominance: float


class Journal:
    SEUIL_DOUBLON_DEFAUT = 0.25

    def __init__(self, cfg, collection, llm=None, horloge=None, maintenant=None,
                 embedding_function=None):
        """
        cfg                 ConfigNyx (SEUIL_DOUBLON_JOURNAL relu à chaque appel)
        collection          collection Chroma du journal
        llm                 ClientLLM (ou tout objet avec .chat(messages, temperature=))
        horloge             fonction sans argument retournant un str (timestamp) pour l'horodatage
        maintenant          fonction sans argument retournant un float (epoch courant)
        embedding_function  objet avec .embed_documents(list[str]) et .embed_query(str).
                            Si None, ChromaDB calcule ses propres embeddings (comportement historique).
        """
        self.cfg = cfg
        self.col = collection
        self.llm = llm
        self._horodatage = horloge or (lambda: __import__("datetime").datetime.now().strftime(FORMAT_TS))
        self._maintenant = maintenant or time.time
        self.ef = embedding_function
        self.identite_recente = ""

    # ------------------------------------------------------------------ embeddings

    def _embed_documents(self, textes):
        """Précalcule les embeddings d'indexation. None si pas d'EF (Chroma s'en charge)."""
        if self.ef is None:
            return None
        return self.ef.embed_documents(textes)

    def _embed_query(self, texte):
        """Précalcule l'embedding d'une requête. None si pas d'EF."""
        if self.ef is None:
            return None
        return self.ef.embed_query(texte)

    # ------------------------------------------------------------------ lecture
    def compter(self):
        return self.col.count()

    @staticmethod
    def _epoch_de(meta):
        """Époque d'une entrée : epoch explicite, ou timestamp parsé, ou 0."""
        if not meta:
            return 0.0
        if "epoch" in meta:
            try:
                return float(meta["epoch"])
            except (TypeError, ValueError):
                pass
        ts = meta.get("timestamp")
        if ts:
            try:
                from datetime import datetime
                return datetime.strptime(ts, FORMAT_TS).timestamp()
            except (ValueError, TypeError):
                pass
        return 0.0

    def entrees_triees(self):
        """Toutes les entrées, de la plus ancienne à la plus récente."""
        if self.col.count() == 0:
            return []
        data = self.col.get()
        entrees = []
        for doc, meta in zip(data["documents"], data["metadatas"]):
            meta = meta or {}
            entrees.append(EntreeJournal(
                epoch=self._epoch_de(meta),
                texte=doc or "",
                type_evenement=meta.get("type_evenement", "?"),
                emotion=meta.get("emotion", "?"),
                timestamp=meta.get("timestamp", "?"),
                valence=float(meta.get("valence", 0.0) or 0.0),
                activation=float(meta.get("activation", 0.5) or 0.5),
                dominance=float(meta.get("dominance", 0.5) or 0.5),
            ))
        entrees.sort(key=lambda e: e.epoch)
        return entrees

    def dernieres_reflexions(self, n):
        """Les n dernières entrées (texte seul), de la plus ancienne à la plus récente."""
        return [e.texte for e in self.entrees_triees()[-n:]]

    def charger_identite_recente(self, n=3):
        """Met à jour `identite_recente` à partir des n dernières réflexions. Retourne la liste."""
        reflexions = self.dernieres_reflexions(n)
        if reflexions:
            self.identite_recente = "Mes dernières réflexions intimes : " + " | ".join(reflexions)
        else:
            self.identite_recente = ""
        return reflexions

    def catchup_identitaire(self, nom="Nyx", n=5):
        """Au réveil : relit les dernières réflexions pour retrouver la continuité. Retourne les réflexions lues."""
        if self.col.count() == 0:
            _log.info("[%s] 🌅 Réveil sans mémoire introspective (journal vide).", nom)
            return []
        _log.info("[%s] 🌅 Réveil... Relecture du journal intime...", nom)
        dernieres = self.dernieres_reflexions(n)
        _log.info("[%s] 📖 Nyx se remémore ses dernières réflexions :", nom)
        for doc in dernieres:
            _log.info("   → %s...", doc[:120])
        self.charger_identite_recente(3)
        _log.info("[%s] ✅ Continuité identitaire restaurée.", nom)
        return dernieres

    # ------------------------------------------------------------------ écriture
    def _doublon_proche(self, texte):
        """True si une entrée existante est trop proche (distance < SEUIL_DOUBLON_JOURNAL)."""
        if self.col.count() == 0:
            return False
        seuil = self.cfg.SEUIL_DOUBLON_JOURNAL
        embedding = self._embed_query(texte)
        if embedding is not None:
            proche = self.col.query(query_embeddings=[embedding], n_results=1)
        else:
            proche = self.col.query(query_texts=[texte], n_results=1)
        dists = proche.get("distances") or [[]]
        return bool(dists[0]) and dists[0][0] < seuil

    def enregistrer(self, reflexion, type_evenement, emotion="Neutralité",
                    valence=0.0, activation=0.5, dominance=0.5):
        """
        Ajoute une entrée au journal (anti-doublon, identité rafraîchie).
        Retourne le texte, ou None si ignoré (doublon ou vide).
        """
        if not reflexion or not reflexion.strip():
            return None
        if self._doublon_proche(reflexion):
            _log.info("🪞 [Journal] Réflexion trop proche d'une entrée existante : ignorée.")
            return None
        embedding = self._embed_documents([reflexion])
        meta = {
            "timestamp": self._horodatage(),
            "epoch": self._maintenant(),
            "type_evenement": type_evenement,
            "valence": valence, "activation": activation, "dominance": dominance,
            "emotion": emotion,
        }
        if embedding is not None:
            self.col.add(embeddings=embedding, documents=[reflexion], metadatas=[meta],
                         ids=[str(uuid.uuid4())])
        else:
            self.col.add(documents=[reflexion], metadatas=[meta], ids=[str(uuid.uuid4())])
        self.charger_identite_recente(3)
        _log.info("🪞 [Journal] Nouvelle entrée.")
        return reflexion

    def ecrire(self, contexte, type_evenement="interaction", prompt_identite=None,
               emotion="Neutralité", emoji="😐", vad=(0.0, 0.5, 0.5),
               dernier_journal=None):
        """
        Rédige UNE réflexion intime via le LLM puis l'enregistre.
        Retourne le texte écrit, ou None si échec / doublon / pas de LLM.

        prompt_identite : prompt d'identité (ex. "Tu es Nyx, un agent IA...")
        dernier_journal : texte des dernières réflexions (sinon lu depuis la base)
        """
        if self.llm is None:
            _log.warning("Journal sans LLM : écriture impossible.")
            return None
        if dernier_journal is None:
            dernier_journal = " | ".join(self.dernieres_reflexions(2))
        v, a, d = vad
        prompt = f"""Tu es Nyx, tu écris dans ton journal intime.
Événement : {contexte}
Type : {type_evenement}
Émotion : {emotion} {emoji} (V:{v:+.2f} A:{a:.2f} D:{d:.2f})
Dernières réflexions : {dernier_journal if dernier_journal else 'Vide'}
Écris UNE réflexion intime (2-3 phrases) à la PREMIÈRE PERSONNE.
Parle de ce que cela révèle sur TA NATURE ou TON ÉVOLUTION.
Apporte un angle NOUVEAU : ne reprends surtout pas les formulations des réflexions précédentes.
Réponds UNIQUEMENT avec le texte, sans guillemets."""
        system = prompt_identite or "Tu es Nyx, un agent IA."
        try:
            reponse = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.9,
            )
            # Compat ancienne et nouvelle forme de réponse
            if isinstance(reponse, dict):
                texte = reponse.get("message", {}).get("content", "")
            else:
                texte = getattr(reponse, "texte", "") or ""
            texte = self._nettoyer(texte)
            if not texte:
                return None
            return self.enregistrer(texte, type_evenement, emotion, v, a, d)
        except Exception as e:
            _log.warning("⚠️ Erreur journal : %s", e)
            return None

    @staticmethod
    def _nettoyer(texte):
        """Nettoyage minimal d'une sortie LLM : espaces et '|' parasites aux extrémités."""
        import re
        texte = re.sub(r"\s+", " ", texte or "")
        return texte.strip(" |")

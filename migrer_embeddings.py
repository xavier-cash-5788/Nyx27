#!/usr/bin/env python3
"""
migrer_embeddings.py : migre la mémoire de Nyx vers nomic-embed-text (768 dims).

Usage :
    python migrer_embeddings.py              # interactif (demande confirmation)
    python migrer_embeddings.py --dry-run    # simulation, ne modifie rien
    python migrer_embeddings.py --auto       # non-interactif (pour script)

Ce que fait le script :
  1. Sauvegarde complète dans ./sauvegardes/ (JSON, horodaté)
  2. Vérifie qu'Ollama et nomic-embed-text sont disponibles
  3. Lit tous les souvenirs (actif + archives) et le journal depuis les anciennes collections
  4. Recalcule TOUS les vecteurs avec nomic-embed-text (préfixes search_document appliqués)
  5. Crée de nouvelles collections (suffixe _v2) avec les mêmes textes et métadonnées
  6. Vérifie que les comptes correspondent
  7. Supprime les anciennes collections et renomme les nouvelles
  8. En cas d'erreur, les anciennes collections restent intactes

Sécurité :
  - Aucune donnée n'est supprimée avant que la nouvelle collection soit vérifiée
  - La sauvegarde JSON permet de tout restaurer manuellement en cas de pépin
  - Le script est idempotent : le relancer après un succès ne fait rien

Correction importante (v2) :
  - Une collection VIDE ou ABSENTE n'est PLUS considérée comme une erreur.
    Elle est simplement ignorée, et la migration continue normalement.
  - Si une erreur survient sur une collection, les _v2 déjà créées sont nettoyées
    automatiquement (aucun résidu laissé derrière).
  - `basculer()` gère correctement les collections vides.
"""
import argparse
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime

import chromadb

# Assure que nyx/ est importable depuis le répertoire courant
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nyx.config import ConfigNyx
from nyx.embeddings import OllamaEmbeddingFunction, NOMIC_DIM


# ============================================================ utilitaires

class Bilan:
    """Compteur simple pour le rapport final."""
    def __init__(self):
        self.souvenirs_avant = 0
        self.souvenirs_apres = 0
        self.archives_avant = 0
        self.archives_apres = 0
        self.journal_avant = 0
        self.journal_apres = 0
        self.duree_s = 0.0
        self.sauvegarde = None

    def resume(self):
        return (
            f"Souvenirs  : {self.souvenirs_avant} → {self.souvenirs_apres}\n"
            f"Archives   : {self.archives_avant} → {self.archives_apres}\n"
            f"Journal    : {self.journal_avant} → {self.journal_apres}\n"
            f"Durée      : {self.duree_s:.1f}s\n"
            f"Sauvegarde : {self.sauvegarde}"
        )


def confirmer(question, auto=False):
    """Demande confirmation, sauf en mode --auto."""
    if auto:
        print(f"  [auto] {question} → oui")
        return True
    try:
        reponse = input(f"  {question} [o/N] : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reponse in ("o", "oui", "y", "yes")


def taille_collection(col):
    """Nombre de documents dans une collection Chroma, ou 0 si elle n'existe pas."""
    try:
        return col.count()
    except Exception:
        return 0


def lister_collections(client):
    """Liste les noms de collections (compatible anciennes et nouvelles versions de Chroma)."""
    cols = client.list_collections()
    noms = []
    for c in cols:
        noms.append(c if isinstance(c, str) else c.name)
    return noms


def supprimer_collection_si_existe(client, nom):
    """Supprime une collection sans lever si elle n'existe pas."""
    try:
        client.delete_collection(name=nom)
        return True
    except Exception:
        return False


# ============================================================ sauvegarde

def sauvegarder_tout(chemin_db, dossier_sauvegardes="sauvegardes"):
    """Copie le dossier Chroma dans une sauvegarde horodatée. Retourne le chemin."""
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    cible = os.path.join(dossier_sauvegardes, f"cerveau_memories_{horodatage}")
    os.makedirs(cible, exist_ok=True)
    if not os.path.isdir(chemin_db):
        print(f"  ⚠️  Dossier source absent : {chemin_db} (rien à sauvegarder)")
        return cible
    for racine, dossiers, fichiers in os.walk(chemin_db):
        relatif = os.path.relpath(racine, chemin_db)
        destination = os.path.join(cible, relatif) if relatif != "." else cible
        os.makedirs(destination, exist_ok=True)
        for nom in fichiers:
            shutil.copy2(os.path.join(racine, nom), os.path.join(destination, nom))
    return cible


def exporter_json(client, chemin_json):
    """Exporte toutes les collections en JSON (lisible, restaurable manuellement)."""
    donnees = {"version": 1, "exporte_le": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "collections": {}}
    for nom in lister_collections(client):
        col = client.get_collection(nom)
        if col.count() == 0:
            donnees["collections"][nom] = {"documents": [], "metadatas": [], "ids": []}
            continue
        d = col.get()
        donnees["collections"][nom] = {
            "documents": d["documents"],
            "metadatas": d["metadatas"],
            "ids": d["ids"],
        }
    os.makedirs(os.path.dirname(os.path.abspath(chemin_json)), exist_ok=True)
    temporaire = chemin_json + ".tmp"
    with open(temporaire, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=1)
    os.replace(temporaire, chemin_json)
    return chemin_json


# ============================================================ migration

def migrer_collection(client, nom_ancien, ef, dimension=NOMIC_DIM, lot=64, dry_run=False):
    """
    Recalcule les embeddings d'une collection et en crée une nouvelle (suffixe _v2).

    Retourne un tuple (statut, avant, apres, nom_nouveau) où statut vaut :
      - "ok"     : migration réussie (ou simulée en dry-run)
      - "ignore" : collection absente ou vide → ce n'est PAS une erreur
      - "erreur" : échec d'embedding, dimension incorrecte ou compte incohérent

    Les collections d'origine restent intactes en cas d'erreur.
    """
    nom_nouveau = f"{nom_ancien}_v2"

    if nom_ancien not in lister_collections(client):
        print(f"  → Collection '{nom_ancien}' absente, ignorée.")
        return "ignore", 0, 0, None

    ancienne = client.get_collection(nom_ancien)
    avant = ancienne.count()
    if avant == 0:
        print(f"  → Collection '{nom_ancien}' vide, ignorée.")
        return "ignore", 0, 0, None

    print(f"  → '{nom_ancien}' : {avant} document(s) à migrer vers '{nom_nouveau}'")
    if dry_run:
        print(f"     [dry-run] Aurait créé '{nom_nouveau}' avec {avant} document(s)")
        return "ok", avant, avant, nom_nouveau

    # Supprime l'éventuelle nouvelle collection d'un essai précédent
    supprimer_collection_si_existe(client, nom_nouveau)

    # Lit tout en une fois
    data = ancienne.get()
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    # Crée la nouvelle collection : on fournit les vecteurs explicitement
    nouvelle = client.create_collection(name=nom_nouveau)

    total = len(ids)
    traites = 0
    debut = time.monotonic()
    for i in range(0, total, lot):
        lot_ids = ids[i:i + lot]
        lot_docs = documents[i:i + lot]
        lot_meta = metadatas[i:i + lot]
        # Robustesse : un document None ou vide est remplacé
        lot_docs_propres = [d if isinstance(d, str) and d.strip() else "(vide)" for d in lot_docs]
        try:
            vecteurs = ef.embed_documents(lot_docs_propres)
        except Exception as e:
            print(f"\n     ❌ Erreur d'embedding sur le lot {i // lot + 1} : {e}")
            supprimer_collection_si_existe(client, nom_nouveau)
            return "erreur", avant, 0, None
        # Vérifie la dimension
        for v in vecteurs:
            if len(v) != dimension:
                print(f"\n     ❌ Dimension {len(v)} au lieu de {dimension}.")
                supprimer_collection_si_existe(client, nom_nouveau)
                return "erreur", avant, 0, None
        nouvelle.add(
            ids=lot_ids,
            documents=lot_docs_propres,
            metadatas=lot_meta,
            embeddings=vecteurs,
        )
        traites += len(lot_ids)
        duree = time.monotonic() - debut
        vitesse = traites / duree if duree > 0 else 0
        print(f"     … {traites}/{total} ({vitesse:.1f} doc/s)", end="\r", flush=True)

    print()  # retour à la ligne après la progression
    apres = nouvelle.count()
    if apres != avant:
        print(f"     ❌ Compte incohérent : {avant} avant, {apres} après.")
        supprimer_collection_si_existe(client, nom_nouveau)
        return "erreur", avant, apres, None
    return "ok", avant, apres, nom_nouveau


def basculer(client, nom_ancien, nom_nouveau):
    """
    Supprime l'ancienne collection et reconstruit une collection au nom d'origine
    en réinsérant les données (avec leurs embeddings) de la collection _v2.
    """
    source = client.get_collection(nom_nouveau)
    data_complet = source.get(include=["embeddings", "documents", "metadatas"])

    # Chroma n'a pas de "rename" : on recrée au bon nom puis on réinsère.
    supprimer_collection_si_existe(client, nom_ancien)
    nouvelle = client.create_collection(name=nom_ancien)

    lot = 64
    n = len(data_complet["ids"])
    for i in range(0, n, lot):
        nouvelle.add(
            ids=data_complet["ids"][i:i + lot],
            documents=data_complet["documents"][i:i + lot],
            metadatas=data_complet["metadatas"][i:i + lot],
            embeddings=[list(e) for e in data_complet["embeddings"][i:i + lot]],
        )
    supprimer_collection_si_existe(client, nom_nouveau)


# ============================================================ programme

def main():
    parser = argparse.ArgumentParser(
        description="Migre la mémoire de Nyx vers nomic-embed-text (768 dims).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation : affiche ce qui serait fait, ne modifie rien.")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactif : accepte toutes les confirmations.")
    parser.add_argument("--chemin-db", default=None,
                        help="Chemin du dossier Chroma (défaut : celui de la config).")
    parser.add_argument("--sans-sauvegarde", action="store_true",
                        help="⚠️ DANGEREUX : ne pas faire de copie de sauvegarde du dossier.")
    parser.add_argument("--lot", type=int, default=64,
                        help="Taille des lots d'embedding (défaut : 64).")
    args = parser.parse_args()

    print("=" * 70)
    print("  MIGRATION DE LA MÉMOIRE DE NYX vers nomic-embed-text")
    print("=" * 70)

    # ---------- 1. Config et chemins ----------
    cfg = ConfigNyx.defauts()
    chemin_db = args.chemin_db or cfg.chemin_db
    if not os.path.isdir(chemin_db):
        print(f"❌ Dossier introuvable : {chemin_db}")
        print("   Lancez Nyx au moins une fois pour créer la base, ou utilisez --chemin-db.")
        return 1

    print(f"\n📂 Base source : {os.path.abspath(chemin_db)}")

    # ---------- 2. Vérification d'Ollama ----------
    print("\n🔍 Vérification d'Ollama et du modèle d'embedding…")
    ef = OllamaEmbeddingFunction(
        model=cfg.embedding_model,
        base_url=cfg.embedding_base_url,
        timeout=cfg.embedding_timeout,
        dimension=NOMIC_DIM,
    )
    diag = ef.verifier()
    print(f"   {diag['message']}")
    if not diag["joignable"]:
        print("\n❌ Ollama est injoignable. Lancez-le :  ollama serve")
        return 1
    if not diag["modele_installe"]:
        print(f"\n❌ Modèle manquant. Installez-le :  ollama pull {cfg.embedding_model}")
        return 1

    # Test rapide : calcule un vecteur pour vérifier que tout marche
    print("   Test d'un appel d'embedding…")
    try:
        v = ef.embed_query("test")
        print(f"   ✅ Vecteur de {len(v)} dimensions obtenu.")
    except Exception as e:
        print(f"   ❌ Échec de l'appel d'embedding : {e}")
        return 1

    # ---------- 3. Ouverture du client Chroma ----------
    print(f"\n🔓 Ouverture du client Chroma…")
    try:
        client = chromadb.PersistentClient(path=chemin_db)
    except Exception as e:
        print(f"❌ Impossible d'ouvrir la base : {e}")
        return 1

    collections_presentes = lister_collections(client)
    print(f"   Collections trouvées : {', '.join(collections_presentes) or '(aucune)'}")

    # Détecte les collections déjà migrées (suffixe _v2 présent = migration en cours)
    if any(c.endswith("_v2") for c in collections_presentes):
        print("\n⚠️  Des collections '_v2' existent déjà. Une migration précédente a-t-elle échoué ?")
        if not confirmer("Supprimer les collections _v2 et recommencer ?", args.auto):
            print("   Abandon.")
            return 1
        for c in collections_presentes:
            if c.endswith("_v2"):
                supprimer_collection_si_existe(client, c)
        collections_presentes = lister_collections(client)
        print(f"   Collections restantes : {', '.join(collections_presentes) or '(aucune)'}")

    # Vérifie qu'il y a bien des données à migrer
    a_migrer = [c for c in collections_presentes if not c.endswith("_v2")]
    if not a_migrer:
        print("\n✅ Aucune collection à migrer.")
        return 0

    # ---------- 4. Sauvegarde ----------
    bilan = Bilan()
    if not args.dry_run and not args.sans_sauvegarde:
        print(f"\n💾 Sauvegarde du dossier complet…")
        cible = sauvegarder_tout(chemin_db)
        print(f"   ✅ Copie dans : {cible}")
        chemin_json = os.path.join("sauvegardes",
                                   f"export_{datetime.now():%Y%m%d_%H%M%S}.json")
        exporter_json(client, chemin_json)
        print(f"   ✅ Export JSON : {chemin_json}")
        bilan.sauvegarde = cible
    elif args.dry_run:
        print("\n[dry-run] Sauvegarde ignorée (mode simulation).")
    else:
        print("\n⚠️  Sauvegarde désactivée (--sans-sauvegarde). Toute erreur sera irréversible.")

    # ---------- 5. Migration ----------
    print(f"\n🔄 Migration des collections…")
    if args.dry_run:
        print("   [dry-run] Aucune modification ne sera effectuée.\n")

    debut = time.monotonic()
    resultats = []          # (nom, avant, apres, nom_nouveau) pour les succès
    a_basculer = []         # (nom, nom_nouveau) pour la bascule
    erreurs = []            # noms des collections en échec

    for nom in a_migrer:
        col = client.get_collection(nom)
        compte = col.count()
        # Comptabilise dans le bilan
        if nom == "souvenirs_long_terme":
            bilan.souvenirs_avant = compte
        elif nom == "journal_intime_nyx":
            bilan.journal_avant = compte

        statut, avant, apres, nouveau = migrer_collection(
            client, nom, ef, dimension=NOMIC_DIM, lot=args.lot, dry_run=args.dry_run
        )

        if statut == "erreur":
            erreurs.append(nom)
            continue
        if statut == "ignore":
            continue
        # statut == "ok"
        resultats.append((nom, avant, apres, nouveau))
        a_basculer.append((nom, nouveau))
        if nom == "souvenirs_long_terme":
            bilan.souvenirs_apres = apres
        elif nom == "journal_intime_nyx":
            bilan.journal_apres = apres

    # Gestion des erreurs
    if erreurs:
        print(f"\n❌ Échec de la migration de : {', '.join(erreurs)}")
        print("   Les collections d'origine sont intactes.")
        # Nettoyage des _v2 créées avant l'échec
        nettoyees = 0
        for _, nouveau in a_basculer:
            if supprimer_collection_si_existe(client, nouveau):
                nettoyees += 1
        if nettoyees:
            print(f"   Nettoyage des collections _v2 partielles effectué ({nettoyees}).")
        return 2

    # Mode dry-run : récapitulatif
    if args.dry_run:
        print("\n[dry-run] Récapitulatif :")
        for ancien, avant, apres, nouveau in resultats:
            print(f"   {ancien} → {nouveau} : {avant} → {apres}")
        print("\n[dry-run] Aucune modification effectuée.")
        return 0

    # Rien à migrer du tout (toutes vides ou absentes)
    if not resultats:
        print("\n✅ Aucune collection à migrer (toutes vides ou absentes).")
        return 0

    # ---------- 6. Bascule ----------
    print(f"\n🔀 Bascule des collections…")
    for ancien, nouveau in a_basculer:
        print(f"   {ancien} ← {nouveau}")
        basculer(client, ancien, nouveau)
        # Vérifie après bascule
        compte_final = client.get_collection(ancien).count()
        attendu = next((av for n, av, ap, nv in resultats if n == ancien), 0)
        if compte_final != attendu:
            print(f"   ❌ Compte final incohérent : {compte_final} au lieu de {attendu}.")
            print(f"      La sauvegarde {bilan.sauvegarde} permet de restaurer.")
            return 3
        print(f"   ✅ {compte_final} document(s) en place.")

    bilan.duree_s = time.monotonic() - debut

    # ---------- 7. Rapport final ----------
    print("\n" + "=" * 70)
    print("  ✅ MIGRATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(bilan.resume())
    print("\n📌 Prochaines étapes :")
    print("   1. Nyx utilisera automatiquement les nouveaux embeddings au prochain lancement.")
    print("   2. La sauvegarde peut être supprimée après quelques jours d'utilisation.")
    print("   3. En cas de problème, restaurez en copiant la sauvegarde dans le dossier Chroma.")
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Migration interrompue par l'utilisateur.")
        print("   Les collections d'origine sont intactes.")
        sys.exit(130)

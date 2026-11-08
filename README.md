# 🌙 Nyx — Agent cérébral à mémoire persistante

> Un agent IA local qui **se souvient**, **ressent**, **rêve** et **écrit**.
> Propulsé par Ollama, ChromaDB et nomic-embed-text.

---

## Table des matières

1. [Présentation](#présentation)
2. [Fonctionnalités](#fonctionnalités)
3. [Installation](#installation)
4. [Premier lancement](#premier-lancement)
5. [Utilisation](#utilisation)
6. [Architecture](#architecture)
7. [Configuration](#configuration)
8. [Embeddings et mémoire sémantique](#embeddings-et-mémoire-sémantique)
9. [Limite de mémoire et purge](#limite-de-mémoire-et-purge)
10. [Migration](#migration)
11. [Diagnostic et dépannage](#diagnostic-et-dépannage)
12. [Tests](#tests)
13. [Structure du projet](#structure-du-projet)
14. [FAQ](#faq)
15. [Licence](#licence)

---

## Présentation

**Nyx** est un agent conversationnel local qui possède :

- Une **mémoire à long terme** persistante (ChromaDB + nomic-embed-text).
- Un **système limbique** (modèle VAD : Valence, Activation, Dominance).
- Un **journal intime** qui forge sa continuité identitaire.
- Des **pulsions** (fatigue, sommeil automatique, pensées spontanées).
- Un **agenda** avec rappels d'événements.
- La capacité d'**écrire des documents** sur demande.
- Une **interface graphique** moderne (customtkinter) et une CLI.

Contrairement à un chatbot classique, Nyx **oublie** ses souvenirs trop faibles, **consolide** les souvenirs neutres en concepts, et **se réveille** avec une identité qui a évolué pendant son sommeil.

---

## Fonctionnalités

### 🧠 Mémoire vivante

- **Oubli actif** : les souvenirs faibles sont archivés au sommeil.
- **Consolidation** : les souvenirs neutres deviennent des concepts généraux.
- **Protection des marquants** : les souvenirs émotionnellement forts ne sont jamais oubliés.
- **Rappel congruent à l'humeur** : quand Nyx est triste, elle se souvient plus facilement des souvenirs tristes.
- **Archive restaurable** : rien n'est jamais détruit (sauf demande explicite).

### ❤️ Émotions

- Modèle VAD : chaque message influence la **valence** (plaisir), l'**activation** (énergie), la **dominance** (contrôle).
- 11 émotions reconnues : joie, peur, colère, tristesse, curiosité, etc.
- Auto-decay : l'humeur revient au neutre avec le temps.
- Feedback 👍/👎 influence directement l'humeur.

### 💭 Pulsions

- **Fatigue** : augmente avec les messages, le temps éveillé, et l'encombrement de souvenirs presque oubliés.
- **Envie de réfléchir** : augmente quand les souvenirs s'estompent (l'érosion nourrit le besoin de penser).
- **Sommeil automatique** : Nyx s'endort d'elle-même quand elle est épuisée et inactive.
- **Pensées spontanées** : Nyx réfléchit à voix haute, ce qui renforce ses souvenirs fragiles.

### 📖 Journal intime

- Chaque événement marquant (feedback, sommeil, document écrit) génère une réflexion à la première personne.
- L'identité de Nyx évolue avec son journal.
- Anti-doublon : les réflexions trop similaires sont ignorées.

### 📅 Agenda

- Ajout d'événements par langage naturel : *« rappelle-moi d'appeler maman demain à 14h »*.
- Rappels automatiques pendant les ticks.
- Agenda du jour intégré dans le prompt système.

### 📄 Documents

- Écriture sur demande : *« écris-moi une note sur la philosophie antique »*.
- Recherche dans les documents existants avant d'écrire.
- Nommage automatique (`notes_de_philo.txt`).

---

## Installation

### Prérequis

- **Python 3.10+**
- **Ollama** installé et lancé ([ollama.com](https://ollama.com))
- **Linux** (testé), **macOS** ou **Windows** (WSL recommandé)
- ~5 Go d'espace disque (modèles Ollama + base Chroma)

### Étapes

```bash
# 1. Cloner ou copier le projet
cd /chemin/vers/nyx

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# ou : venv\Scripts\activate   # Windows

# 3. Installer les dépendances Python
pip install --upgrade pip
pip install chromadb ollama customtkinter pytest

# 4. Télécharger les modèles Ollama
ollama pull qwen2.5:7b                  # Modèle principal (chat, journal, émotions)
ollama pull nomic-embed-text:latest     # Modèle d'embeddings (mémoire sémantique)

# 5. Lancer Ollama (si pas déjà en service)
ollama serve &
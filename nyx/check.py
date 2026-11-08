"""
nyx/check.py : auto-diagnostic global de Nyx (module 9/9).
Remplace la méthode check() de l'ancien main.py.

Toutes les écritures se font dans une SANDBOX (base temporaire) :
la vraie mémoire n'est jamais modifiée (la base réelle n'est lue qu'en lecture seule).
Le rapport détaillé est écrit dans ./logs/nyx_check_<date>.txt

Ce module utilise le nouvel AgentCerebralNyx modulaire (nyx.agent) pour créer la sandbox,
garantissant la parité totale avec les tests existants.
"""
import builtins
import contextlib
import gc
import io
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta

from .agent import AgentCerebralNyx
from .llm import ReponseLLM

FORMAT_TS = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------------ outils de mock LLM
def _poser_llm(t, faux):
    """Installe un faux chat sur le client LLM partagé : le pipeline et tous les modules l'utilisent."""
    t.llm.chat = faux  # attribut d'instance qui masque la méthode ClientLLM.chat


def _retirer_llm(t):
    """Retire le faux : t.llm.chat redevient la vraie méthode ClientLLM.chat."""
    t.llm.__dict__.pop("chat", None)


def _reponse_llm(contenu, modele="faux"):
    """Construit une réponse compatible avec tout ce que les modules attendent."""
    return ReponseLLM(contenu, modele, 0.0, 0.0, 1)


def _faux_llm(reponse_texte="Réponse simulée.",
              reponse_emotion='{"valence": 0.5, "activation": 0.6, "dominance": 0.5}'):
    """Faux chat : format='json' -> émotion, sinon -> texte ; n'appelle jamais Ollama."""
    def faux(messages, format=None, temperature=None, modele=None, timeout=None, **kw):
        contenu = reponse_emotion if format == "json" else reponse_texte
        return _reponse_llm(contenu)
    return faux


def _llm_en_panne(*a, **kw):
    raise ConnectionError("Ollama simulé en panne")


def run_check(agent, rapide=False):
    """
    Auto-diagnostic global. Toutes les écritures se font dans une SANDBOX (base temporaire) :
    la vraie mémoire n'est jamais modifiée (la base réelle n'est lue qu'en lecture seule).
    Le rapport détaillé est écrit dans ./logs/nyx_check_<date>.txt

    agent : l'agent Nyx (nouveau agent modulaire) dont on veut faire le diagnostic
    rapide : si True, saute les tests qui nécessitent le modèle
    """
    debut = time.time()
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    chemin_log = os.path.join("logs", f"nyx_check_{horodatage}.txt")
    suffixe = 2
    while os.path.exists(chemin_log):
        chemin_log = os.path.join("logs", f"nyx_check_{horodatage}_{suffixe}.txt")
        suffixe += 1

    dossier_tmp = tempfile.mkdtemp(prefix="nyx_check_")
    ctx = {'llm_ok': True}
    resultats = []
    icones = {"OK": "✅", "AVERT": "⚠️ ", "ÉCHEC": "❌", "ERREUR": "💥", "SKIP": "⏭️ "}

    print(f"\n🔎 === AUTO-DIAGNOSTIC DE {agent.nom.upper()} ({'rapide' if rapide else 'complet'}) ===")
    print("   Les tests d'écriture utilisent une base temporaire : votre mémoire réelle n'est pas modifiée.\n")

    # ---------- utilitaires ----------
    def executer(nom, fn, llm=False, sandbox=True):
        raison = None
        if llm and rapide:
            raison = "mode rapide"
        elif llm and not ctx['llm_ok']:
            raison = "Ollama ou modèle indisponible"
        elif sandbox and 't' not in ctx:
            raison = "sandbox non créée"
        if raison:
            resultats.append({'nom': nom, 'statut': 'SKIP', 'duree': 0.0, 'detail': raison, 'sortie': ''})
            print(f"  {icones['SKIP']} {nom} (ignoré : {raison})")
            return
        tampon = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(tampon):
                retour = fn()
            if isinstance(retour, tuple):
                statut, detail = retour
            else:
                statut, detail = "OK", retour or ""
        except AssertionError as e:
            statut, detail = "ÉCHEC", str(e) or "assertion échouée"
        except Exception:
            statut, detail = "ERREUR", traceback.format_exc()
        duree = time.time() - t0
        resultats.append({'nom': nom, 'statut': statut, 'duree': duree,
                          'detail': detail, 'sortie': tampon.getvalue().strip()})
        print(f"  {icones[statut]} {nom} ({duree:.1f}s)")

    def il_y_a(heures):
        return (datetime.now() - timedelta(hours=heures)).strftime(FORMAT_TS)

    def fabriquer_meta(dernier=None, force=0.6, v=0.0, a=0.5, typ="souvenir_brut"):
        ts = agent._obtenir_timestamp_actuel() if hasattr(agent, '_obtenir_timestamp_actuel') else datetime.now().strftime(FORMAT_TS)
        return {"created_at": ts, "dernier_acces": dernier or ts, "force_base": force,
                "reactivations": 0, "type": typ, "valence": v, "activation": a, "dominance": 0.5}

    def ajouter(t, doc, **kw):
        t.hippocampe.add(documents=[doc], metadatas=[fabriquer_meta(**kw)], ids=[str(uuid.uuid4())])

    def vider(col):
        ids = col.get()['ids']
        if ids:
            col.delete(ids=ids)

    def reinitialiser_emotions(t):
        t.valence, t.activation, t.dominance = 0.0, 0.5, 0.5

    # ---------- 1. environnement & base réelle ----------
    def t_env():
        import importlib.metadata as md
        def version(pkg):
            try:
                return md.version(pkg)
            except Exception:
                return "?"
        return "\n".join([
            f"Python {sys.version.split()[0]} sur {platform.platform()}",
            f"chromadb {version('chromadb')} | ollama (lib) {version('ollama')}",
            f"Modèle chat : {agent.model_name} | consolidation : {agent.model_consolidation} | num_ctx : {agent.num_ctx}",
            f"Réflexion préfrontale : {agent.reflexion_prefrontale}",
            f"Seuils : oubli={agent.SEUIL_OUBLI} distance={agent.SEUIL_DISTANCE} "
            f"doublon_journal={agent.SEUIL_DOUBLON_JOURNAL} decay={agent.TAUX_DECAY}/h",
            f"Base réelle : {os.path.abspath(agent.chemin_db)}",
        ])

    def t_ollama():
        try:
            installes = agent._lister_modeles_ollama() if hasattr(agent, '_lister_modeles_ollama') else agent.llm.lister_modeles()
        except Exception as e:
            ctx['llm_ok'] = False
            raise AssertionError(f"Ollama injoignable : {e}. Lancez 'ollama serve'.")
        manquants = [mod for mod in sorted({agent.model_name, agent.model_consolidation})
                     if not (hasattr(agent, '_modele_installe') and agent._modele_installe(mod, installes))]
        if manquants:
            ctx['llm_ok'] = False
            raise AssertionError(
                f"Modèle(s) non installé(s) : {', '.join(manquants)} (faites 'ollama pull <modèle>'). "
                f"Installés : {', '.join(installes) or 'aucun'}")
        return f"Ollama joignable. Modèles installés : {', '.join(installes)}"

    def t_base_reelle():
        lignes = [f"hippocampe : {agent.hippocampe.count()} | renforcement : "
                  f"{agent.memoire_renforcement.count()} | journal : {agent.journal_intime.count()}"]
        problemes, forces = [], []
        if agent.hippocampe.count():
            data = agent.hippocampe.get()
            for doc_id, doc, meta in zip(data['ids'], data['documents'], data['metadatas']):
                court = doc_id[:8]
                if meta is None:
                    problemes.append(f"{court} : métadonnées absentes (sera oublié au prochain sommeil)")
                    continue
                try:
                    mm = agent._normaliser_meta(meta) if hasattr(agent, '_normaliser_meta') else agent.memoire.normaliser(meta)
                    datetime.strptime(mm['dernier_acces'], FORMAT_TS)
                    datetime.strptime(mm['created_at'], FORMAT_TS)
                except (ValueError, TypeError) as e:
                    problemes.append(f"{court} : métadonnées invalides ({e})")
                    continue
                if not 0.0 <= mm['force_base'] <= 1.0:
                    problemes.append(f"{court} : force_base hors [0,1] ({mm['force_base']})")
                if not doc or not doc.strip():
                    problemes.append(f"{court} : document vide")
                if mm['type'] == 'concept_consolidé' and not (hasattr(agent, '_concept_valide') and agent._concept_valide(doc)):
                    problemes.append(f"{court} : concept suspect « {doc[:60]}... » (commande 'nettoyer')")
                forces.append(agent._force_effective(meta) if hasattr(agent, '_force_effective') else agent.memoire.force_effective(meta))
        if forces:
            lignes.append(f"force effective : moyenne {sum(forces) / len(forces):.2f} "
                          f"(min {min(forces):.2f}, max {max(forces):.2f})")
        if problemes:
            return "AVERT", "\n".join(lignes + ["Problèmes :"] + [f"  - {p}" for p in problemes])
        return "\n".join(lignes)

    def t_sandbox():
        ctx['t'] = AgentCerebralNyx(
            model_name=agent.model_name, utilisateur=agent.utilisateur, num_ctx=agent.num_ctx,
            reflexion_prefrontale=agent.reflexion_prefrontale,
            model_consolidation=agent.model_consolidation, chemin_db=dossier_tmp)
        t = ctx['t']
        assert t.hippocampe.count() == 0 and t.journal_intime.count() == 0, "sandbox non vide"
        return f"Sandbox créée : {dossier_tmp}"

    # ---------- 2. logique interne (sans modèle) ----------
    def t_temps_et_nettoyage():
        t = ctx['t']
        assert t._heures_depuis(t._obtenir_timestamp_actuel()) < 0.01, "_heures_depuis(maintenant) != 0"
        assert t._heures_depuis("pas une date") == 0.0, "date invalide non gérée"
        assert t._heures_depuis(None) == 0.0, "None non géré"
        assert t._nettoyer_entree("  bonjour   Nyx \n") == "bonjour Nyx"
        assert t._nettoyer_entree("J'ai un iPhone") == "J'ai un iPhone", "la casse est altérée"
        assert t._nettoyer_sortie_llm(" | a   b | ") == "a b"
        return "timestamps, nettoyage d'entrée et de sortie OK"

    def t_decay():
        t = ctx['t']
        neutre = {"force_base": 0.6, "dernier_acces": il_y_a(10), "created_at": il_y_a(10)}
        f1 = [t._force_effective(neutre) for _ in range(5)]
        assert max(f1) - min(f1) < 1e-6, "decay non idempotent (il se cumule à chaque appel)"
        attendu = 0.6 * math.exp(-t.TAUX_DECAY * 10)
        assert abs(f1[0] - attendu) < 0.01, f"force à 10h = {f1[0]:.3f}, attendu {attendu:.3f}"
        assert t._force_effective(dict(neutre, dernier_acces=il_y_a(0))) > f1[0], "un souvenir récent devrait être plus fort"
        marquant = dict(neutre, valence=-0.9, activation=0.9)
        assert t._force_effective(marquant) > f1[0], "un souvenir marquant devrait décroître plus lentement"
        ancien = {"force_synaptique": 0.6, "timestamp": il_y_a(10)}
        assert abs(t._force_effective(ancien) - f1[0]) < 0.01, "ancien schéma de métadonnées mal lu"
        norm = t._normaliser_meta(None)
        assert norm['type'] == 'souvenir_brut' and 0 <= norm['force_base'] <= 1
        charge = t._calculer_charge_emotionnelle({"valence": -0.9, "activation": 0.9})
        assert abs(charge - 0.81) < 1e-6 and t._est_marquant({"valence": -0.9, "activation": 0.9})
        assert not t._est_marquant({"valence": 0.1, "activation": 0.5})
        return f"force à 10h = {f1[0]:.3f} (attendu {attendu:.3f}) ; marquant = {t._force_effective(marquant):.3f}"

    def t_emotions():
        t = ctx['t']
        noms = set()
        for v in (-1, -0.5, 0, 0.5, 1):
            for a in (0, 0.5, 1):
                for d in (0, 0.5, 1):
                    t.valence, t.activation, t.dominance = v, a, d
                    nom, emoji = t._calculer_emotion_dominante()
                    assert nom and emoji, f"émotion vide pour V={v} A={a} D={d}"
                    assert t._generer_instruction_emotionnelle_concise(nom, emoji), f"pas d'instruction pour « {nom} »"
                    noms.add(nom)
        reinitialiser_emotions(t)
        for stim in ((1.0, 1.0, 1.0), (-1.0, 0.0, 0.0)):
            for _ in range(200):
                t._mettre_a_jour_emotions(*stim)
            assert -1 <= t.valence <= 1 and 0 <= t.activation <= 1 and 0 <= t.dominance <= 1, \
                f"émotions hors bornes : {t.valence}, {t.activation}, {t.dominance}"
        reinitialiser_emotions(t)
        return f"{len(noms)} émotions atteignables : {', '.join(sorted(noms))}"

    def t_concepts():
        t = ctx['t']
        rejetes = [
            "Voici la consolidation de mes souvenirs neutres : - Mon rôle dans l'alchimie",
            "Bien sûr, je suis ravi de vous aider ! Voici mes réponses",
            "L'alchimie est une affinité. **Concept 2 : Le temps** Le temps nous coince...",
            "Je fusionne les concepts suivants : **Consolidation**",
            "trop court", "x" * 300, "ligne 1\nligne 2 assez longue pour passer", 42, None,
        ]
        acceptes = [
            "L'introspection critique consiste à revoir ses propres biais pour améliorer ses réponses.",
            "La Limite Épistémique : la conscience d'être limité par ses connaissances.",
        ]
        for r in rejetes:
            assert not t._concept_valide(r), f"devrait être rejeté : {str(r)[:60]}"
        for a in acceptes:
            assert t._concept_valide(a), f"devrait être accepté : {a[:60]}"
        return f"{len(rejetes)} rejets et {len(acceptes)} acceptations conformes"

    def t_rappel():
        t = ctx['t']
        textes = ["J'adore cuisiner des pâtes italiennes le dimanche",
                  "Mon chien s'appelle Rex et il aime courir",
                  "La philosophie de Descartes repose sur le doute"]
        for txt in textes:
            ajouter(t, txt)
        assert t.hippocampe.count() == 3
        ids, docs, _ = t._rappeler_souvenirs(textes[0])
        assert textes[0] in docs, "un texte identique n'est pas rappelé"
        d_proche = t.hippocampe.query(query_texts=["Quelle est ta recette de pâtes ?"], n_results=1)['distances'][0][0]
        d_loin = t.hippocampe.query(query_texts=["Explique la physique quantique des trous noirs"], n_results=1)['distances'][0][0]
        id0 = ids[docs.index(textes[0])]
        avant = t.hippocampe.get(ids=[id0])['metadatas'][0]
        t._renforcer_souvenirs([id0])
        apres = t.hippocampe.get(ids=[id0])['metadatas'][0]
        assert apres['reactivations'] == avant.get('reactivations', 0) + 1, "réactivations non incrémentées"
        assert apres['force_base'] > avant['force_base'], "force non renforcée"
        for _ in range(10):
            t._renforcer_souvenirs([id0])
        assert t.hippocampe.get(ids=[id0])['metadatas'][0]['force_base'] <= 1.0, "force > 1.0 après renforcements"
        t._renforcer_souvenirs([])
        vider(t.hippocampe)
        detail = (f"distance rappel identique = 0, sujet proche = {d_proche:.2f}, "
                  f"sujet sans rapport = {d_loin:.2f} (seuil actuel {t.SEUIL_DISTANCE})")
        avert = []
        if d_proche > t.SEUIL_DISTANCE:
            avert.append("seuil trop STRICT : un sujet proche serait ignoré → augmentez SEUIL_DISTANCE")
        if d_loin <= t.SEUIL_DISTANCE:
            avert.append("seuil trop PERMISSIF : un sujet sans rapport serait rappelé → diminuez SEUIL_DISTANCE")
        return ("AVERT", detail + "\n" + "\n".join(avert)) if avert else detail

    def t_journal_tri():
        t = ctx['t']
        base = time.time()
        for nom, decalage in (("C", 3), ("A", 1), ("B", 2)):
            t.journal_intime.add(
                documents=[f"réflexion {nom}"],
                metadatas=[{"timestamp": t._obtenir_timestamp_actuel(), "epoch": base + decalage,
                            "type_evenement": "test", "emotion": "Neutralité",
                            "valence": 0.0, "activation": 0.5, "dominance": 0.5}],
                ids=[str(uuid.uuid4())])
        assert t._dernieres_reflexions(2) == ["réflexion B", "réflexion C"], \
            f"dernières réflexions incorrectes : {t._dernieres_reflexions(2)}"
        assert [d for _, d, _ in t._entrees_journal_triees()] == ["réflexion A", "réflexion B", "réflexion C"]
        t._charger_identite_recente(2)
        assert "réflexion B" in t.identite_recente and "réflexion C" in t.identite_recente
        assert "réflexion A" not in t.identite_recente
        t.journal_intime.add(documents=["ancienne entrée"], metadatas=[{"timestamp": "2000-01-01 00:00:00"}],
                             ids=[str(uuid.uuid4())])
        assert t._dernieres_reflexions(1) == ["réflexion C"], "entrée à l'ancien format mal triée"
        vider(t.journal_intime)
        t.identite_recente = ""
        return "tri par date OK (y compris entrées à l'ancien format sans 'epoch')"

    def t_prompt_historique():
        t = ctx['t']
        t.identite_recente = "Mes dernières réflexions intimes : test"
        t._construire_prompt_systeme("SOUVENIR_X", ["FEEDBACK_Y"], "PLAN_Z", "2000-01-01 00:00:00")
        prompt = t.memoire_de_travail[0]['content']
        for attendu in ("SOUVENIR_X", "FEEDBACK_Y", "PLAN_Z", "2000-01-01", "Ton actuel"):
            assert attendu in prompt, f"« {attendu} » absent du prompt système"
        systeme = t.memoire_de_travail[0]
        t.memoire_de_travail = [systeme] + [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f"m{i}"} for i in range(40)]
        t._tronquer_historique()
        assert len(t.memoire_de_travail) <= t.MAX_HISTORIQUE + 1, "historique non tronqué"
        assert t.memoire_de_travail[0]['role'] == 'system', "prompt système perdu"
        assert t.memoire_de_travail[1]['role'] == 'user', "l'historique ne commence pas par 'user'"
        t.memoire_de_travail = [systeme]
        t.identite_recente = ""
        return f"prompt système complet ; historique tronqué à {t.MAX_HISTORIQUE} messages"

    def t_dormir_sans_llm():
        t = ctx['t']
        # Silence total du LLM pour journal et consolidation
        llm_journal, llm_conso = t.journal.llm, t.consolidation.llm
        t.journal.llm = t.consolidation.llm = None
        try:
            t.dormir()  # base vide : ne doit pas planter
            textes = {"oublie": "Souvenir banal très ancien à oublier",
                      "marquant": "Souvenir très émotionnel ancien à préserver",
                      "concept": "Concept consolidé ancien à préserver",
                      "recent": "Souvenir récent neutre à garder"}
            ajouter(t, textes["oublie"], dernier=il_y_a(100), force=0.5)
            ajouter(t, textes["marquant"], dernier=il_y_a(100), force=0.5, v=-0.9, a=0.9)
            ajouter(t, textes["concept"], dernier=il_y_a(30), force=0.95, typ="concept_consolidé")
            ajouter(t, textes["recent"])
            t.dormir()
            restants = set(t.hippocampe.get()['documents'])
            assert textes["oublie"] not in restants, "le souvenir faible n'a pas été oublié"
            assert textes["marquant"] in restants, "un souvenir marquant a été oublié"
            assert textes["concept"] in restants, "un concept consolidé récent a été oublié"
            assert textes["recent"] in restants, "un souvenir récent a été oublié"
            vider(t.hippocampe)
            reinitialiser_emotions(t)
        finally:
            t.journal.llm, t.consolidation.llm = llm_journal, llm_conso
        return "oubli actif, protection des marquants et des concepts OK"

    def t_affichages():
        t = ctx['t']
        fonctions = (("afficher_memoire", t.afficher_memoire), ("afficher_emotions", t.afficher_emotions),
                     ("etat_cerebral", t.etat_cerebral), ("afficher_journal", t.afficher_journal),
                     ("se_decrire (journal vide)", lambda: print(t.se_decrire())),
                     ("afficher_aide", t.afficher_aide))
        lignes = {}
        def lancer(phase):
            for nom, fn in fonctions:
                if phase == "rempli" and nom.startswith("se_decrire"):
                    continue
                tampon = io.StringIO()
                with contextlib.redirect_stdout(tampon):
                    fn()
                sortie = tampon.getvalue()
                assert sortie.strip(), f"{nom}() n'affiche rien ({phase})"
                lignes[f"{nom}/{phase}"] = len(sortie.splitlines())
        lancer("vide")
        ajouter(t, "Un souvenir pour l'affichage")
        ajouter(t, "Un concept pour l'affichage", typ="concept_consolidé")
        ajouter(t, "Un souvenir marquant pour l'affichage", v=-0.9, a=0.9)
        t.journal_intime.add(documents=["entrée de test"], metadatas=[{"timestamp": t._obtenir_timestamp_actuel(),
                             "epoch": time.time(), "type_evenement": "test", "emotion": "Neutralité"}],
                             ids=[str(uuid.uuid4())])
        lancer("rempli")
        vider(t.hippocampe)
        vider(t.journal_intime)
        return f"{len(lignes)} affichages OK (base vide et remplie)"

    def t_nettoyer():
        t = ctx['t']
        pollue = "**Concept 1 : Bla** Voici une réponse polluée du modèle qui ne devrait pas être là"
        bon = "L'introspection consiste à examiner ses propres biais de raisonnement."
        ajouter(t, pollue, typ="concept_consolidé")
        ajouter(t, bon, typ="concept_consolidé")
        ancien_input = builtins.input
        builtins.input = lambda *a, **k: 'o'
        try:
            t.nettoyer_concepts()
        finally:
            builtins.input = ancien_input
        restants = t.hippocampe.get()['documents']
        assert pollue not in restants, "le concept pollué n'a pas été supprimé"
        assert bon in restants, "un concept valide a été supprimé"
        vider(t.hippocampe)
        return "suppression ciblée des concepts pollués OK"

    def t_feedback():
        t = ctx['t']
        llm_journal = t.journal.llm
        t.journal.llm = None
        try:
            t.derniere_reponse = None
            t.donner_feedback(1)
            assert t.memoire_renforcement.count() == 0, "feedback enregistré sans réponse préalable"
            t.dernier_stimulus = "Parle-moi de la cuisine italienne"
            t.derniere_reponse = "Les pâtes fraîches sont la base de la cuisine italienne."
            v0 = t.valence
            t.donner_feedback(1)
            assert t.memoire_renforcement.count() == 1 and t.valence > v0, "feedback positif sans effet"
            v1 = t.valence
            t.donner_feedback(-1)
            assert t.memoire_renforcement.count() == 2 and t.valence < v1, "feedback négatif sans effet"
            retours = t._rappeler_feedbacks("Parle-moi de la cuisine italienne")
            assert retours, "les feedbacks passés ne sont pas retrouvés"
            assert any("appréciée" in r or "mal reçue" in r for r in retours)
        finally:
            t.journal.llm = llm_journal
            t.derniere_reponse = t.dernier_stimulus = None
            reinitialiser_emotions(t)
        return f"feedbacks enregistrés et rappelés : {retours}"

    def t_pipeline_simule():
        t = ctx['t']
        _poser_llm(t, _faux_llm("Réponse simulée."))
        try:
            avant = t.hippocampe.count()
            r1 = t.percevoir_et_repondre("Ceci est un message de test unique")
            assert r1 == "Réponse simulée.", f"réponse inattendue : {r1}"
            assert [nom for nom, _ in t.dernier_pipeline] == [
                "Signal", "Filtre émotionnel", "Décodage mnésique", "Décision", "Comportement"], \
                f"étapes du pipeline inattendues : {t.dernier_pipeline}"
            assert all(duree >= 0 for _, duree in t.dernier_pipeline)
            assert t.charge_msgs >= t.FATIGUE_PAR_MESSAGE - 1e-9, "un échange ne fatigue pas Nyx"
            assert t.hippocampe.count() == avant + 1, "message non mémorisé"
            assert t.percevoir_et_repondre("   ") == "", "entrée vide mal gérée"
            t.percevoir_et_repondre("Ceci est un message de test unique")
            assert t.hippocampe.count() == avant + 1, "doublon exact mémorisé deux fois"
            _poser_llm(t, _llm_en_panne)
            n_hist, n_mem = len(t.memoire_de_travail), t.hippocampe.count()
            r3 = t.percevoir_et_repondre("Message envoyé pendant une panne")
            assert r3.startswith("[Erreur"), f"panne LLM non signalée : {r3}"
            assert len(t.memoire_de_travail) == n_hist, "l'historique est pollué après une panne LLM"
            assert t.hippocampe.count() == n_mem, "un message a été mémorisé malgré la panne LLM"
            assert len(t.dernier_pipeline) == 4, "après une panne, le pipeline doit s'arrêter avant « Comportement »"
        finally:
            _retirer_llm(t)
            t.charge_msgs = 0.0
            vider(t.hippocampe)
            t.memoire_de_travail = [t.memoire_de_travail[0]]
            t.derniere_reponse = t.dernier_stimulus = None
            reinitialiser_emotions(t)
        return "pipeline complet avec modèle simulé : mémorisation, doublons, panne LLM OK"

    def t_rappel_humeur():
        t = ctx['t']
        texte = "Je repense à mon grand projet de livre"
        ajouter(t, texte, v=-0.8, a=0.8)
        ajouter(t, texte, v=0.8, a=0.8)
        ajouter(t, "La physique quantique et les trous noirs supermassifs", v=-0.8, a=0.8)
        try:
            t.activation = 0.8
            t.valence = -0.9
            _, docs_neg, metas_neg = t._rappeler_souvenirs(texte)
            assert len(metas_neg) >= 2, "les deux souvenirs pertinents devraient être rappelés"
            assert float(metas_neg[0]['valence']) < 0, "humeur négative : le souvenir négatif devrait passer en premier"
            t.valence = 0.9
            _, _, metas_pos = t._rappeler_souvenirs(texte)
            assert float(metas_pos[0]['valence']) > 0, "humeur positive : le souvenir positif devrait passer en premier"
            t.valence = 0.0
            _, docs_neutre, _ = t._rappeler_souvenirs(texte)
            assert len(docs_neutre) >= 2, "humeur neutre : le rappel doit fonctionner sans biais"
            _, docs_off, _ = t._rappeler_souvenirs(texte, humeur=False)
            assert len(docs_off) >= 2
            hors_sujet = any("trous noirs" in d for d in docs_neg)
        finally:
            reinitialiser_emotions(t)
            vider(t.hippocampe)
        detail = ("humeur négative → souvenir négatif en premier ; humeur positive → souvenir positif en premier ; "
                  "humeur neutre → aucun biais")
        if hors_sujet:
            return "AVERT", detail + "\nUn souvenir sans rapport a été rappelé : SEUIL_DISTANCE est trop permissif."
        return detail

    def t_derive_emotions():
        t = ctx['t']
        try:
            t.valence, t.activation, t.dominance = 0.8, 0.9, 0.1
            t._deriver_emotions(t.TAU_EMOTION)
            assert abs(t.valence - 0.8 * math.exp(-1)) < 1e-6, f"valence après τ : {t.valence}"
            assert 0.5 < t.activation < 0.9 and 0.1 < t.dominance < 0.5, "l'humeur ne retombe pas vers le neutre"
            avant = (t.valence, t.activation, t.dominance)
            t._deriver_emotions(0)
            t._deriver_emotions(-5)
            assert (t.valence, t.activation, t.dominance) == avant, "un dt nul ou négatif ne doit rien changer"
            t._deriver_emotions(10 ** 6)
            assert abs(t.valence) < 1e-6 and abs(t.activation - 0.5) < 1e-6 and abs(t.dominance - 0.5) < 1e-6
        finally:
            reinitialiser_emotions(t)
        return "l'humeur retombe vers le neutre avec le temps (auto-decay émotionnel)"

    def t_fatigue_sommeil_auto():
        t = ctx['t']
        # Silence journal + consolidation : dormir() va être appelé dans ce test
        llm_journal, llm_conso = t.journal.llm, t.consolidation.llm
        t.journal.llm = t.consolidation.llm = None
        evenements = []
        t.sur_evenement = lambda genre, texte: evenements.append(genre)
        t0 = time.time()

        def remise_a_zero(charge=0.0):
            t.debut_eveil = t.derniere_interaction = t.derniere_auto_reflexion = t.dernier_tick = t0
            t.charge_msgs = charge
            t.envie_dormir_annoncee = False

        try:
            t.reflexion_auto_actif = False
            t.sommeil_auto_actif = True
            remise_a_zero()
            base = t.calculer_pulsions(t0)["fatigue"]
            assert base < 0.05, f"fatigue initiale anormale : {base:.2f}"
            assert t.calculer_pulsions(t0 + 3 * 3600)["fatigue"] > base + 0.2, "le temps éveillé ne fatigue pas"
            remise_a_zero(0.5)
            assert abs(t.calculer_pulsions(t0)["fatigue"] - 0.5) < 0.02, "les messages ne fatiguent pas"
            remise_a_zero()
            for i in range(4):
                ajouter(t, f"Souvenir presque oublié {i}", dernier=il_y_a(100), force=0.5)
            assert t.calculer_pulsions(t0)["fatigue"] >= 0.25, "l'encombrement de souvenirs presque oubliés ne fatigue pas"
            vider(t.hippocampe)
            remise_a_zero(0.65)
            assert t.tick(t0) == "envie_dormir", "Nyx devrait dire qu'elle a sommeil"
            assert evenements == ["envie_dormir"]
            assert t.tick(t0 + 1) is None, "l'envie de dormir ne doit être annoncée qu'une fois"
            t.charge_msgs = 0.95
            assert t.tick(t0 + 10) is None, "Nyx s'est endormie en pleine conversation"
            assert t.tick(t0 + 60) == "dormir", "Nyx devrait s'endormir après une période d'inactivité"
            assert evenements[-2:] == ["sommeil", "reveil"]
            assert t.charge_msgs == 0.0 and not t.envie_dormir_annoncee, "l'état n'est pas remis à zéro au réveil"
            assert t.calculer_pulsions()["fatigue"] < t.SEUIL_ENVIE_DORMIR, "la fatigue n'est pas retombée après le sommeil"
            t.sommeil_auto_actif = False
            remise_a_zero(0.95)
            t.envie_dormir_annoncee = True
            assert t.tick(t0 + 600) is None, "sommeil automatique alors qu'il est désactivé"
        finally:
            t.sur_evenement = None
            t.sommeil_auto_actif = t.reflexion_auto_actif = True
            t.charge_msgs = 0.0
            vider(t.hippocampe)
            vider(t.journal_intime)
            reinitialiser_emotions(t)
            t.journal.llm, t.consolidation.llm = llm_journal, llm_conso
        return (f"fatigue ↑ avec les messages, le temps et l'encombrement ; annonce à ≥ {t.SEUIL_ENVIE_DORMIR}, "
                f"sommeil automatique à ≥ {t.SEUIL_SOMMEIL_AUTO} seulement si inactive")

    def t_auto_reflexion_decay():
        t = ctx['t']
        evenements = []
        t.sur_evenement = lambda genre, texte: evenements.append(genre)
        envie_par_minute = t.ENVIE_PAR_MINUTE
        t0 = time.time()
        _poser_llm(t, _faux_llm(
            "Je repense à ces souvenirs qui s'estompent, et je veux les garder vivants en moi."))
        try:
            t.sommeil_auto_actif = t.reflexion_auto_actif = True
            t.debut_eveil = t.derniere_interaction = t.derniere_auto_reflexion = t.dernier_tick = t0
            t.charge_msgs = 0.0
            t.envie_dormir_annoncee = False
            for i in range(3):
                ajouter(t, f"Souvenir qui s'estompe numéro {i}", dernier=il_y_a(14), force=0.8)
            ajouter(t, "Souvenir tout frais", force=0.9)
            p0 = t.calculer_pulsions(t0)
            assert p0["nb_declin"] == 3, f"3 souvenirs devraient s'estomper, trouvé {p0['nb_declin']}"
            assert p0["envie_reflexion"] < t.SEUIL_REFLEXION
            assert t.tick(t0 + 100) is None, "pensée spontanée trop précoce"
            p1 = t.calculer_pulsions(t0 + 15 * 60)
            assert p1["envie_reflexion"] >= t.SEUIL_REFLEXION, "avec le temps, l'envie de penser devrait monter"
            assert t.tick(t0 + 15 * 60) == "auto_reflexion", "Nyx devrait penser à voix haute"
            assert evenements.count("auto_reflexion") == 1
            metas = t.hippocampe.get()['metadatas']
            assert sum(1 for m in metas if int(m.get('reactivations', 0)) >= 1) >= 2, \
                "la pensée n'a pas renforcé les souvenirs visés"
            assert any(m.get('type_evenement') == 'auto_reflexion' for m in t.journal_intime.get()['metadatas']), \
                "la pensée n'a pas été écrite au journal"
            assert t.derniere_pensee and "s'estompent" in t.derniere_pensee[1]
            p2 = t.calculer_pulsions(t0 + 15 * 60 + 5)
            assert p2["nb_declin"] < p0["nb_declin"] and p2["envie_reflexion"] < p1["envie_reflexion"], \
                "la répétition devrait freiner l'érosion et calmer l'envie"
            t.ENVIE_PAR_MINUTE = 100.0
            assert t.tick(t0 + 15 * 60 + 10) is None, "deux pensées spontanées trop rapprochées"
            t.ENVIE_PAR_MINUTE = envie_par_minute
            _poser_llm(t, _llm_en_panne)
            t.derniere_auto_reflexion = t.derniere_interaction = t0
            assert t.tick(t0 + 2 * 3600) is None
            assert t.derniere_auto_reflexion == t0 + 2 * 3600, "pas d'anti-boucle après une panne du modèle"
            _poser_llm(t, _faux_llm(
                "Je repense à ces souvenirs qui s'estompent, et je veux les garder vivants en moi."))
            t.reflexion_auto_actif = False
            t.envie_dormir_annoncee = True
            t.derniere_auto_reflexion = t0
            assert t.tick(t0 + 3 * 3600) is None, "pensée spontanée alors qu'elles sont désactivées"
        finally:
            _retirer_llm(t)
            t.sur_evenement = None
            t.ENVIE_PAR_MINUTE = envie_par_minute
            t.sommeil_auto_actif = t.reflexion_auto_actif = True
            t.charge_msgs = 0.0
            t.derniere_pensee = None
            vider(t.hippocampe)
            vider(t.journal_intime)
            reinitialiser_emotions(t)
        return "l'érosion de la mémoire fait monter l'envie de penser ; la pensée renforce les souvenirs et calme l'envie"

    # ---------- 3. avec le vrai modèle ----------
    def t_llm_ping():
        t = ctx['t']
        lignes = []
        for modele in sorted({t.model_name, t.model_consolidation}):
            rep = t._appeler_llm([{'role': 'user', 'content': "Réponds uniquement par le mot OK."}],
                                 temperature=0, modele=modele)
            texte = rep['message']['content'].strip()
            assert texte, f"réponse vide de {modele}"
            lignes.append(f"{modele} → « {texte[:60]} »")
        return "\n".join(lignes)

    def t_llm_emotion():
        t = ctx['t']
        heureux = t._analyser_stimulus_emotionnel("Je suis tellement heureux, quelle magnifique journée !")
        furieux = t._analyser_stimulus_emotionnel("Je suis furieux, tout est nul et personne ne m'écoute.")
        for v, a, d in (heureux, furieux):
            assert -1 <= v <= 1 and 0 <= a <= 1 and 0 <= d <= 1, "valeurs hors bornes"
        detail = f"message heureux → V={heureux[0]:+.2f} A={heureux[1]:.2f} D={heureux[2]:.2f}\n" \
                 f"message furieux → V={furieux[0]:+.2f} A={furieux[1]:.2f} D={furieux[2]:.2f}"
        if (0.0, 0.5, 0.5) in (heureux, furieux):
            return "AVERT", detail + "\nValeurs par défaut : repli probable (le modèle ne répond pas en JSON conforme)."
        if heureux[0] <= furieux[0]:
            return "AVERT", detail + "\nLa valence du message heureux n'est pas supérieure à celle du message furieux."
        return detail

    def t_llm_conversation():
        t = ctx['t']
        avant = t.hippocampe.count()
        texte = "Bonjour Nyx, je m'appelle Xavier et j'adore la philosophie antique."
        rep = t.percevoir_et_repondre(texte)
        assert rep and not rep.startswith("[Erreur"), f"réponse invalide : {rep[:150]}"
        assert t.hippocampe.count() == avant + 1, "le message n'a pas été mémorisé"
        assert t.derniere_reponse == rep and t.dernier_stimulus == texte
        roles = [msg['role'] for msg in t.memoire_de_travail]
        assert roles[0] == 'system' and roles[-2:] == ['user', 'assistant'], f"historique incohérent : {roles}"
        notes = []
        if t.reflexion_prefrontale and "Plan interne" not in t.memoire_de_travail[0]['content']:
            notes.append("le plan préfrontal n'apparaît pas dans le prompt système")
        _, docs, _ = t._rappeler_souvenirs("philosophie antique")
        if texte not in docs:
            notes.append("le souvenir n'est pas rappelé pour « philosophie antique » (SEUIL_DISTANCE trop strict ?)")
        detail = f"Nyx a répondu : « {rep[:200]} »"
        return ("AVERT", detail + "\n" + "\n".join(notes)) if notes else detail

    def t_llm_journal():
        t = ctx['t']
        avant = t.journal_intime.count()
        refl = t._ecrire_dans_journal("Test de vérification : je traverse un auto-diagnostic complet.", "check")
        if not refl:
            return "AVERT", "réflexion vide, ignorée (doublon) ou erreur modèle : voir la sortie console ci-dessous"
        assert t.journal_intime.count() == avant + 1, "entrée non enregistrée"
        assert refl in t.identite_recente, "l'identité n'a pas été rafraîchie"
        return f"Entrée écrite : « {refl[:150]} »"

    def t_llm_dormir():
        t = ctx['t']
        vider(t.hippocampe)
        sujets = ["J'aime les balades en forêt le week-end", "Mon café préféré est l'espresso serré",
                  "Je lis un roman policier chaque soir", "Je fais du vélo le samedi matin"]
        for s in sujets:
            ajouter(t, s)
        t.dormir()
        data = t.hippocampe.get()
        concepts = [d for d, meta in zip(data['documents'], data['metadatas'])
                    if meta and meta.get('type') == 'concept_consolidé']
        assert t.hippocampe.count() >= 1, "perte totale des souvenirs pendant le sommeil"
        if not concepts:
            assert t.hippocampe.count() == len(sujets), "souvenirs perdus sans concept créé"
            return ("AVERT", "aucun concept valide produit par le modèle : souvenirs conservés "
                             "(la sécurité fonctionne, mais la consolidation est inopérante avec ce modèle)")
        for c in concepts:
            assert t._concept_valide(c), f"concept invalide enregistré : {c[:80]}"
        return f"{len(sujets)} souvenirs → {len(concepts)} concept(s) : " + " | ".join(c[:90] for c in concepts)

    def t_llm_identite():
        t = ctx['t']
        tampon = io.StringIO()
        with contextlib.redirect_stdout(tampon):
            print(t.se_decrire())
        sortie = tampon.getvalue().strip()
        assert len(sortie) > 40 and "Erreur" not in sortie, f"introspection invalide : {sortie[:150]}"
        fuites = [mot for mot in ("alibaba", "openai", "qwen", "anthropic", "assistant intelligent créé")
                  if mot in sortie.lower()]
        if fuites:
            return "AVERT", (f"Nyx se présente comme le modèle de base (mots repérés : {', '.join(fuites)}).\n"
                             f"{sortie[:600]}")
        return sortie[:600]

    def t_llm_auto_reflexion():
        t = ctx['t']
        ajouter(t, "J'aime discuter de philosophie antique avec Xavier", dernier=il_y_a(14), force=0.8)
        ajouter(t, "Xavier m'a dit qu'il aime les balades en forêt", dernier=il_y_a(14), force=0.8)
        avant = sum(int(m.get('reactivations', 0)) for m in t.hippocampe.get()['metadatas'])
        pensee = t.auto_reflexion(force=True)
        assert pensee, "aucune pensée produite (réponse vide ou invalide : voir la sortie console)"
        assert t._pensee_valide(pensee)
        assert t.derniere_pensee and t.derniere_pensee[1] == pensee
        apres = sum(int(m.get('reactivations', 0)) for m in t.hippocampe.get()['metadatas'])
        assert apres > avant, "la pensée n'a pas renforcé les souvenirs visés"
        return f"Nyx pense à voix haute : « {pensee[:250]} »"

    # ---------- exécution ----------
    print("📦 Environnement")
    executer("Versions et configuration", t_env, sandbox=False)
    executer("Ollama et modèles installés", t_ollama, sandbox=False)
    executer("Intégrité de la base réelle (lecture seule)", t_base_reelle, sandbox=False)
    executer("Création de la sandbox", t_sandbox, sandbox=False)

    print("\n🧩 Logique interne (sans modèle)")
    executer("Temps et nettoyage de texte", t_temps_et_nettoyage)
    executer("Decay, force synaptique, charge émotionnelle", t_decay)
    executer("Système limbique", t_emotions)
    executer("Validation des concepts", t_concepts)
    executer("Stockage, rappel, renforcement, calibrage distances", t_rappel)
    executer("Journal trié par date", t_journal_tri)
    executer("Prompt système et historique", t_prompt_historique)
    executer("Sommeil : oubli et protections", t_dormir_sans_llm)
    executer("Commandes d'affichage", t_affichages)
    executer("Nettoyage des concepts pollués", t_nettoyer)
    executer("Feedback et renforcement", t_feedback)
    executer("Pipeline de conversation (modèle simulé, panne LLM)", t_pipeline_simule)
    executer("Rappel congruent à l'humeur", t_rappel_humeur)
    executer("Auto-decay de l'humeur", t_derive_emotions)
    executer("Fatigue et sommeil automatique", t_fatigue_sommeil_auto)
    executer("Auto-decay mémoire → pensée à voix haute", t_auto_reflexion_decay)

    print("\n🤖 Avec le vrai modèle" + ("" if rapide else " (peut prendre 1 à 3 minutes)"))
    executer("Appel modèle(s)", t_llm_ping, llm=True)
    executer("Analyse émotionnelle", t_llm_emotion, llm=True)
    executer("Conversation complète", t_llm_conversation, llm=True)
    executer("Écriture du journal", t_llm_journal, llm=True)
    executer("Pensée à voix haute", t_llm_auto_reflexion, llm=True)
    executer("Sommeil et consolidation", t_llm_dormir, llm=True)
    executer("Introspection (identite)", t_llm_identite, llm=True)

    # ---------- nettoyage & rapport ----------
    ctx.pop('t', None)
    gc.collect()
    shutil.rmtree(dossier_tmp, ignore_errors=True)
    compte = {s: sum(1 for r in resultats if r['statut'] == s) for s in ("OK", "AVERT", "ÉCHEC", "ERREUR", "SKIP")}
    if compte["ÉCHEC"] or compte["ERREUR"]:
        verdict = "❌ PROBLÈMES DÉTECTÉS : consultez le détail des tests en échec"
    elif compte["AVERT"]:
        verdict = "⚠️  FONCTIONNE, AVEC DES AVERTISSEMENTS (voir le détail)"
    else:
        verdict = "✅ TOUT FONCTIONNE"
    duree_totale = time.time() - debut
    lignes = [
        "=" * 70,
        f"RAPPORT D'AUTO-DIAGNOSTIC DE {agent.nom.upper()}",
        "=" * 70,
        f"Date  : {datetime.now().strftime(FORMAT_TS)}",
        f"Mode  : {'rapide (sans modèle)' if rapide else 'complet'}",
        f"Durée : {duree_totale:.1f}s",
        "",
        "RÉSUMÉ",
        "-" * 70,
    ]
    for r in resultats:
        lignes.append(f"[{r['statut']:<6}] {r['nom']} ({r['duree']:.1f}s)")
    lignes += ["", "DÉTAILS", "-" * 70]
    for r in resultats:
        lignes.append(f"\n### [{r['statut']}] {r['nom']} ({r['duree']:.1f}s)")
        if r['detail']:
            lignes.append(str(r['detail']))
        if r['sortie']:
            lignes.append("--- sortie console capturée ---")
            lignes.append(r['sortie'])
    lignes += ["", "=" * 70,
               f"BILAN : {compte['OK']} OK | {compte['AVERT']} avertissement(s) | "
               f"{compte['ÉCHEC']} échec(s) | {compte['ERREUR']} erreur(s) | {compte['SKIP']} ignoré(s)",
               f"VERDICT : {verdict}", "=" * 70]
    try:
        with open(chemin_log, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes) + "\n")
        emplacement = f"📝 Rapport détaillé : {os.path.abspath(chemin_log)}"
    except OSError as e:
        emplacement = f"⚠️ Impossible d'écrire le rapport : {e}"
    print(f"\n{verdict}")
    print(f"   {compte['OK']} OK | {compte['AVERT']} avertissement(s) | {compte['ÉCHEC']} échec(s) | "
          f"{compte['ERREUR']} erreur(s) | {compte['SKIP']} ignoré(s) | {duree_totale:.1f}s")
    print(f"{emplacement}\n")
    return chemin_log

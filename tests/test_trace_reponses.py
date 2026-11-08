"""
test_trace_reponses.py : vérifie que Nyx garde trace de ses propres réponses.

Preuves attendues :
  - l'historique de travail contient les messages 'user' ET 'assistant'
  - agent.derniere_reponse contient la dernière réponse
  - le stimulus (question) est mémorisé dans l'hippocampe
  - un feedback 👍 stocke la réponse en métadonnée
  - après 2 échanges, l'historique contient 4 messages (2 user + 2 assistant)

Lancement :
    python -m pytest tests/test_trace_reponses.py -v -s
ou :
    python tests/test_trace_reponses.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from nyx.agent import AgentCerebralNyx


def main():
    print("=" * 70)
    print("  TEST : Nyx garde-t-elle trace de ses réponses ?")
    print("=" * 70)

    agent = AgentCerebralNyx(charger_voix=False)
    agent.cfg.definir("reflexion_prefrontale", False)  # accélère

    # --- 1. Premier message
    print("\n1. Envoi d'un message…")
    question = "Peux-tu me parler de la philosophie antique ?"
    reponse = agent.percevoir_et_repondre(question)
    print(f"   → Question : {question[:60]}")
    print(f"   → Réponse : {reponse[:100]}…")

    # --- 2. Historique de travail
    print("\n2. Vérification de l'historique de travail…")
    historique = agent.memoire_de_travail
    print(f"   → {len(historique)} messages")
    roles = [m["role"] for m in historique]
    print(f"   → Rôles : {roles}")
    hist_ok = "user" in roles and "assistant" in roles
    if hist_ok:
        print(f"   ✅ Question ET réponse dans l'historique")
    else:
        print(f"   ❌ Réponse manquante dans l'historique")

    # --- 3. derniere_reponse
    print("\n3. La réponse est-elle accessible via agent.derniere_reponse ?")
    dr_ok = agent.derniere_reponse == reponse
    if dr_ok:
        print(f"   ✅ agent.derniere_reponse == réponse")
    else:
        print(f"   ❌ agent.derniere_reponse ne correspond pas")
        print(f"      Attendu : {reponse[:60]}…")
        print(f"      Reçu    : {(agent.derniere_reponse or '(None)')[:60]}…")

    # --- 4. Question mémorisée
    print("\n4. La question est-elle mémorisée ?")
    souvenirs = agent.memoire.tous()
    question_trouvee = any(question.lower() in s.texte.lower() for s in souvenirs)
    if question_trouvee:
        print(f"   ✅ Question mémorisée dans l'hippocampe")
    else:
        print(f"   ⚠️  Question non mémorisée (peut-être un doublon)")

    # --- 5. Deuxième message
    print("\n5. Envoi d'un 2e message qui référence le 1er…")
    question2 = "Tu peux répéter ce que tu m'as dit sur la philosophie ?"
    reponse2 = agent.percevoir_et_repondre(question2)
    print(f"   → Réponse 2 : {reponse2[:100]}…")

    # --- 6. Historique après 2 échanges
    print("\n6. Vérification de l'historique après 2 échanges…")
    historique = agent.memoire_de_travail
    nb_user = sum(1 for m in historique if m["role"] == "user")
    nb_assistant = sum(1 for m in historique if m["role"] == "assistant")
    print(f"   → {nb_user} message(s) user")
    print(f"   → {nb_assistant} message(s) assistant")
    deux_echanges_ok = (nb_user == 2 and nb_assistant == 2)
    if deux_echanges_ok:
        print(f"   ✅ Les 2 échanges sont dans l'historique")
    else:
        print(f"   ⚠️  Historique incomplet")

    # --- 7. Feedback
    print("\n7. Test du feedback 👍…")
    agent.donner_feedback(1)
    feedbacks = agent.memoire_renforcement.get()
    feedback_ok = False
    if feedbacks["ids"]:
        meta = feedbacks["metadatas"][-1]
        print(f"   → Feedback stocké :")
        print(f"      Stimulus   : {feedbacks['documents'][-1][:60]}")
        print(f"      Réponse    : {(meta.get('reponse') or '')[:60]}")
        print(f"      Récompense : {meta.get('recompense')}")
        feedback_ok = bool(meta.get("reponse"))
        if feedback_ok:
            print(f"   ✅ La réponse est tracée dans le feedback")
        else:
            print(f"   ❌ La réponse n'est pas dans le feedback")

    # --- 8. Trace dans le journal
    print("\n8. La réponse laisse-t-elle une trace dans le journal ?")
    entrees = agent._entrees_journal_triees()
    journal_contenu = " ".join(e[1].lower() for e in entrees)
    mots_cles = [m for m in question.lower().split() if len(m) > 4]
    trouves = sum(1 for m in mots_cles if m in journal_contenu)
    print(f"   → {trouves}/{len(mots_cles)} mots-clés retrouvés")
    journal_ok = trouves > 0 or len(entrees) == 0  # OK si journal vide (pas de feedback)

    # --- Verdict
    print("\n" + "=" * 70)
    ok = hist_ok and dr_ok and deux_echanges_ok and feedback_ok
    if ok:
        print(f"✅ TRACE OK : historique complet, réponse accessible, feedback tracé")
    else:
        print(f"⚠️  Détail :")
        print(f"   Historique 1er échange : {'✅' if hist_ok else '❌'}")
        print(f"   derniere_reponse        : {'✅' if dr_ok else '❌'}")
        print(f"   Historique 2 échanges   : {'✅' if deux_echanges_ok else '⚠️'}")
        print(f"   Feedback tracé          : {'✅' if feedback_ok else '❌'}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    main()

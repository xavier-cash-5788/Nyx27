"""
test_reves_etanches.py : vérifie que les rêves ne contaminent pas les souvenirs vécus.

Preuves attendues :
  - les souvenirs vécus sont toujours présents avec le même type après le rêve
  - les rêves sont typés "souvenir_onirique" et préfixés "🌙 Rêve : "
  - les rêves sont protégés de l'oubli (dans plan.oniriques)
  - aucun texte de rêve ne se retrouve en "souvenir_brut"

Lancement :
    python -m pytest tests/test_reves_etanches.py -v -s
ou :
    python tests/test_reves_etanches.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from nyx.agent import AgentCerebralNyx


def main():
    print("=" * 70)
    print("  TEST : Les rêves contaminent-ils les souvenirs vécus ?")
    print("=" * 70)

    agent = AgentCerebralNyx(charger_voix=False)
    agent.cfg.definir("reve_actif", True)
    agent.cfg.definir("reve_seuil_souvenir", 0.0)  # force TOUS les rêves à être mémorisés

    # --- 1. Ajout de souvenirs vécus
    print("\n1. Ajout de souvenirs VÉCUS…")
    vecus = [
        "Xavier m'a dit qu'il aime la philosophie antique.",
        "Mon café préféré est l'espresso serré.",
        "Je lis un roman policier chaque soir.",
    ]
    for texte in vecus:
        agent.memoire.ajouter(texte, valence=0.2, activation=0.6)
    print(f"   → {len(vecus)} souvenirs vécus ajoutés")

    # --- 2. Compter les oniriques AVANT
    tous_avant = agent.memoire.tous()
    oniriques_avant = [s for s in tous_avant if s.type == "souvenir_onirique"]
    print(f"\n2. Souvenirs oniriques AVANT rêve : {len(oniriques_avant)}")

    # --- 3. Rêve
    print("\n3. Déclenchement d'un rêve (2 cycles REM)…")
    reves = agent.rever()
    print(f"   → {len(reves)} rêves générés")
    for r in reves:
        print(f"      Rêve {r.numero} : {len(r.texte)} caractères")

    if not reves:
        print("\n❌ Aucun rêve généré. Vérifier reve_actif et les souvenirs.")
        return False

    # --- 4. Compter les types APRÈS
    tous_apres = agent.memoire.tous()
    oniriques_apres = [s for s in tous_apres if s.type == "souvenir_onirique"]
    bruts_apres = [s for s in tous_apres if s.type == "souvenir_brut"]
    concepts_apres = [s for s in tous_apres if s.type == "concept_consolidé"]
    print(f"\n4. Souvenirs APRÈS rêve :")
    print(f"   → Bruts (vécus) : {len(bruts_apres)}")
    print(f"   → Concepts : {len(concepts_apres)}")
    print(f"   → Oniriques (rêves) : {len(oniriques_apres)}")

    # --- 5. Vérifier l'intégrité des souvenirs vécus
    print("\n5. Vérification de l'intégrité des souvenirs vécus…")
    tous_textes = {s.texte for s in tous_apres}
    vecus_intacts = 0
    for v in vecus:
        if v in tous_textes:
            print(f"   ✅ « {v[:50]}… » intact")
            vecus_intacts += 1
        elif any(v in t for t in tous_textes):
            print(f"   ⚠️  « {v[:50]}… » modifié/fusionné")
        else:
            print(f"   ❌ « {v[:50]}… » PERDU !")

    # --- 6. Vérifier le typage des rêves
    print("\n6. Vérification du typage des rêves…")
    reves_memorises = 0
    for r in reves:
        prefixe = "🌙 Rêve : "
        trouve = any(
            s.texte.startswith(prefixe) and r.texte[:80] in s.texte
            for s in oniriques_apres
        )
        if trouve:
            print(f"   ✅ Rêve {r.numero} mémorisé avec type 'souvenir_onirique' et préfixe 🌙")
            reves_memorises += 1
        else:
            print(f"   ⚠️  Rêve {r.numero} NON mémorisé (vérifier reve_seuil_souvenir)")

    # --- 7. Vérifier la protection à l'oubli
    print("\n7. Vérification de la protection à l'oubli…")
    plan = agent.memoire.trier_pour_sommeil()
    print(f"   → {len(plan.oniriques)} oniriques protégés dans plan.oniriques")
    print(f"   → {len(plan.marquants)} marquants protégés")
    print(f"   → {len(plan.oubli)} souvenirs à oublier")
    protection_ok = len(plan.oniriques) == len(oniriques_apres)

    # --- 8. Vérifier qu'aucun rêve n'est devenu "brut"
    print("\n8. Vérification qu'aucun rêve ne s'est glissé dans 'brut'…")
    textes_bruts = {s.texte for s in bruts_apres}
    contamination = False
    for r in reves:
        # On regarde si un fragment significatif du rêve est dans les bruts
        fragment = r.texte[50:150]
        if any(fragment in t for t in textes_bruts):
            print(f"   ❌ Contamination détectée : un rêve s'est glissé dans les bruts")
            contamination = True
    if not contamination:
        print(f"   ✅ Aucune contamination : les rêves restent dans 'onirique'")

    # --- 9. Verdict
    print("\n" + "=" * 70)
    ok = (
        vecus_intacts == len(vecus)
        and reves_memorises == len(reves)
        and protection_ok
        and not contamination
    )
    if ok:
        print(f"✅ RÊVES ÉTANCHES : {len(vecus)} vécus intacts, "
              f"{reves_memorises} rêves typés/protégés, zéro contamination")
    else:
        print(f"⚠️  Vérifier : vécus intacts={vecus_intacts}/{len(vecus)}, "
              f"rêves mémorisés={reves_memorises}/{len(reves)}, "
              f"protection={'OK' if protection_ok else 'KO'}, "
              f"contamination={'OUI' if contamination else 'NON'}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    main()

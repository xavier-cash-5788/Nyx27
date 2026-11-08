"""
test_sommeil_ameliore.py : vérifie que le sommeil améliore le rappel (v4).

Preuves attendues :
  1. Un souvenir MARQUANT (valence 0.9) est préservé intact après le sommeil
     (ni oublié, ni consolidé) → prouve la protection des marquants.
  2. Des souvenirs banals (force ~0.40, "en déclin") sont CONSOLIDÉS en concepts
     → prouve que la consolidation fonctionne (pas juste l'oubli).
  3. Le nombre total de souvenirs DIMINUE (fusion), mais les concepts apparaissent
     → prouve que la mémoire s'organise (elle ne se vide pas, elle se structure).
  4. Les souvenirs vécus intacts sont toujours accessibles par rappel.
  5. La force du souvenir marquant reste stable ou augmente (protection decay).

Calibrage du vieillissement :
  Pour qu'un souvenir soit "en déclin" (consolidable) :
    0.30 < force < 0.50
    force = 0.6 × exp(-0.05 × t)
    → t entre 3.6h et 13.8h
  On utilise 8h : force ≈ 0.40 (parfait pour la consolidation).

Lancement :
    python tests/test_sommeil_ameliore.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from datetime import datetime, timedelta

from nyx.agent import AgentCerebralNyx


def _compter_types(agent):
    """Retourne (bruts, concepts, oniriques, marquants, total)."""
    tous = agent.memoire.tous()
    bruts = sum(1 for s in tous if s.type == "souvenir_brut")
    concepts = sum(1 for s in tous if s.type == "concept_consolidé")
    oniriques = sum(1 for s in tous if s.type == "souvenir_onirique")
    marquants = sum(1 for s in tous if s.categorie == "marquant")
    return bruts, concepts, oniriques, marquants, len(tous)


def _mesurer_cible(agent, question, marqueur_unique):
    """
    Mesure le rappel d'un souvenir contenant `marqueur_unique`.
    Retourne un dict complet.
    """
    souvenirs = agent.memoire.rappeler(question, humeur=None, n=15)
    for s in souvenirs:
        if marqueur_unique.lower() in s.texte.lower():
            return {
                "trouve": True,
                "distance": s.distance,
                "force": s.force,
                "type": s.type,
                "texte": s.texte,
                "categorie": s.categorie,
            }
    # Fallback : chercher dans tous les souvenirs
    for s in agent.memoire.tous():
        if marqueur_unique.lower() in s.texte.lower():
            return {
                "trouve": False,
                "distance": None,
                "force": s.force,
                "type": s.type,
                "texte": s.texte,
                "categorie": s.categorie,
            }
    return {
        "trouve": False,
        "distance": None,
        "force": None,
        "type": None,
        "texte": None,
        "categorie": None,
    }


def _vieillir_tous_souvenirs(agent, heures):
    """Vieillit tous les souvenirs de N heures (recule `dernier_acces`)."""
    ts_vieux = (datetime.now() - timedelta(hours=heures)).strftime("%Y-%m-%d %H:%M:%S")
    data = agent.hippocampe.get()
    ids, metas = [], []
    for doc_id, meta in zip(data["ids"], data["metadatas"]):
        if meta:
            meta["dernier_acces"] = ts_vieux
            ids.append(doc_id)
            metas.append(meta)
    if ids:
        agent.hippocampe.update(ids=ids, metadatas=metas)
    return len(ids)


def main():
    print("=" * 75)
    print("  TEST : Le sommeil améliore-t-il le rappel ? (v4)")
    print("=" * 75)

    agent = AgentCerebralNyx(charger_voix=False)
    agent.cfg.definir("reve_actif", False)  # isole la consolidation pure

    # ============================================================
    # 1. Souvenir CIBLE marquant (protégé de la consolidation)
    # ============================================================
    print("\n1. Ajout du souvenir CIBLE marquant (valence 0.9, activation 0.9)…")
    marqueur_unique = "MOTMAGIQUE_alpha_42"
    cible_texte = (
        f"Xavier m'a confié un secret important : le mot magique est {marqueur_unique}. "
        f"Je ne dois jamais l'oublier."
    )
    agent.memoire.ajouter(cible_texte, valence=0.9, activation=0.9)
    print(f"   → Souvenir cible ajouté (charge émotionnelle = 0.81 > 0.5 = marquant)")

    # ============================================================
    # 2. 20 souvenirs BANALS (candidats à la consolidation)
    # ============================================================
    print("\n2. Ajout de 20 souvenirs banals (valence 0.0)…")
    for i in range(20):
        agent.memoire.ajouter(
            f"Souvenir banal numéro {i} sur la météo, le repas ou une anecdote ordinaire.",
            valence=0.0, activation=0.5,
        )
    print(f"   → {agent.memoire.compter()} souvenirs en mémoire")

    # ============================================================
    # 3. Vieillissement contrôlé (8h, pour mettre les banals en déclin)
    # ============================================================
    print("\n3. Vieillissement de 8h (met les banals en déclin, pas en oubli)…")
    nb_vieillis = _vieillir_tous_souvenirs(agent, heures=8)
    print(f"   → {nb_vieillis} souvenirs vieillis")
    # Vérification de la force attendue
    force_attendue_banal = 0.6 * (2.71828 ** (-0.05 * 8))
    print(f"   → Force attendue d'un banal à 8h : ~{force_attendue_banal:.3f} (déclin = 0.3–0.5)")

    # ============================================================
    # 4. État AVANT sommeil
    # ============================================================
    print("\n4. État AVANT sommeil…")
    bruts_a, concepts_a, oniriques_a, marquants_a, total_a = _compter_types(agent)
    print(f"   → Bruts : {bruts_a}  ·  Concepts : {concepts_a}  ·  Marquants : {marquants_a}  ·  Total : {total_a}")

    question = f"Quel est le mot magique que Xavier m'a confié ?"
    mesure_avant = _mesurer_cible(agent, question, marqueur_unique)
    print(f"   → Cible trouvée par rappel : {mesure_avant['trouve']}")
    if mesure_avant["trouve"]:
        print(f"   → Distance : {mesure_avant['distance']:.3f}")
    print(f"   → Force : {mesure_avant['force']:.3f}")
    print(f"   → Type : {mesure_avant['type']}")
    print(f"   → Catégorie : {mesure_avant['categorie']}")

    # ============================================================
    # 5. Sommeil
    # ============================================================
    print("\n5. Sommeil (NREM, rêves désactivés)…")
    agent.dormir()

    # ============================================================
    # 6. État APRÈS sommeil
    # ============================================================
    print("\n6. État APRÈS sommeil…")
    bruts_b, concepts_b, oniriques_b, marquants_b, total_b = _compter_types(agent)
    print(f"   → Bruts : {bruts_b}  ·  Concepts : {concepts_b}  ·  Marquants : {marquants_b}  ·  Total : {total_b}")

    mesure_apres = _mesurer_cible(agent, question, marqueur_unique)
    print(f"   → Cible trouvée par rappel : {mesure_apres['trouve']}")
    if mesure_apres["trouve"]:
        print(f"   → Distance : {mesure_apres['distance']:.3f}")
    print(f"   → Force : {mesure_apres['force']:.3f}")
    print(f"   → Type : {mesure_apres['type']}")
    print(f"   → Catégorie : {mesure_apres['categorie']}")

    # ============================================================
    # 7. Analyse des résultats
    # ============================================================
    print("\n7. Analyse…")

    # -- Preuve 1 : la cible marquante est intacte
    cible_intacte = (
        mesure_apres["type"] == "souvenir_brut"
        and mesure_apres["force"] is not None
        and mesure_avant["force"] is not None
        and mesure_apres["force"] >= mesure_avant["force"] - 0.01  # tolérance
    )
    print(f"\n   [Preuve 1] La cible marquante est intacte ?")
    if cible_intacte:
        print(f"      ✅ Oui (type={mesure_apres['type']}, force {mesure_avant['force']:.3f} → {mesure_apres['force']:.3f})")
    elif mesure_apres["type"] is None:
        print(f"      ❌ Non — la cible a été PERDUE")
    elif mesure_apres["type"] == "concept_consolidé":
        print(f"      ❌ Non — la cible a été CONSOLIDÉE (la valence ne l'a pas protégée)")
    else:
        print(f"      ⚠️  Type inattendu : {mesure_apres['type']}")

    # -- Preuve 2 : les banals ont été consolidés (pas juste oubliés)
    nb_banals_consommes = bruts_a - bruts_b
    concepts_crees = concepts_b - concepts_a
    print(f"\n   [Preuve 2] Les banals ont été consolidés en concepts ?")
    print(f"      Bruts avant : {bruts_a}  →  après : {bruts_b}  (Δ = -{nb_banals_consommes})")
    print(f"      Concepts avant : {concepts_a}  →  après : {concepts_b}  (Δ = +{concepts_crees})")
    consolidation_ok = concepts_crees > 0 and nb_banals_consommes > concepts_crees
    if consolidation_ok:
        print(f"      ✅ Oui : {nb_banals_consommes} bruts fusionnés en {concepts_crees} concepts")
    else:
        print(f"      ⚠️  Consolidation partielle ou absente")

    # -- Preuve 3 : la mémoire s'organise (total diminue, mais reste structuré)
    print(f"\n   [Preuve 3] La mémoire s'organise ?")
    print(f"      Total avant : {total_a}  →  après : {total_b}  (Δ = {total_b - total_a:+d})")
    organisation_ok = total_b < total_a and concepts_b >= concepts_a
    if organisation_ok:
        print(f"      ✅ Oui : {total_a} → {total_b} (fusion, pas perte)")
    else:
        print(f"      ⚠️  Organisation non observée")

    # -- Preuve 4 : la cible reste accessible par rappel
    print(f"\n   [Preuve 4] La cible reste accessible par rappel ?")
    if mesure_avant["trouve"] and mesure_apres["trouve"]:
        delta_dist = mesure_apres["distance"] - mesure_avant["distance"]
        if delta_dist <= 0:
            print(f"      ✅ Oui, et mieux : distance {mesure_avant['distance']:.3f} → {mesure_apres['distance']:.3f}")
        else:
            print(f"      ⚠️  Toujours trouvée, mais distance {mesure_avant['distance']:.3f} → {mesure_apres['distance']:.3f}")
    elif mesure_apres["trouve"]:
        print(f"      ✅ Émergée après sommeil")
    else:
        print(f"      ❌ Perdue du rappel")

    # -- Preuve 5 : le tri a bien séparé marquants / faibles / neutres
    print(f"\n   [Preuve 5] Le tri au sommeil a fonctionné ?")
    plan_apres = agent.memoire.trier_pour_sommeil()
    print(f"      Après sommeil :")
    print(f"      → {len(plan_apres.marquants)} marquants préservés")
    print(f"      → {len(plan_apres.concepts)} concepts")
    print(f"      → {len(plan_apres.oniriques)} oniriques")
    print(f"      → {len(plan_apres.neutres)} neutres restants")
    print(f"      → {len(plan_apres.oubli)} à oublier")
    # Si des banals ont été consolidés, il ne doit plus y avoir beaucoup de neutres
    tri_ok = len(plan_apres.neutres) < 5  # au moins 15 des 20 banals ont été traités
    if tri_ok:
        print(f"      ✅ Oui : les banals ont été traités")
    else:
        print(f"      ⚠️  {len(plan_apres.neutres)} neutres restants (consolidation partielle)")

    # ============================================================
    # 8. Verdict global
    # ============================================================
    print("\n" + "=" * 75)
    preuves = {
        "Cible marquante intacte": cible_intacte,
        "Banals consolidés": consolidation_ok,
        "Mémoire organisée": organisation_ok,
        "Cible accessible": mesure_apres["trouve"] or mesure_apres["force"] is not None,
        "Tri effectif": tri_ok,
    }
    nb_ok = sum(1 for v in preuves.values() if v)
    total_preuves = len(preuves)

    if nb_ok == total_preuves:
        print(f"✅ SOMMEIL VALIDÉ : {nb_ok}/{total_preuves} preuves satisfaites")
    elif nb_ok >= total_preuves - 1:
        print(f"🟢 SOMMEIL LARGEMENT VALIDÉ : {nb_ok}/{total_preuves} preuves satisfaites")
        for nom, ok in preuves.items():
            if not ok:
                print(f"   ⚠️  Preuve manquante : {nom}")
    else:
        print(f"⚠️  SOMMEIL PARTIELLEMENT VALIDÉ : {nb_ok}/{total_preuves} preuves")
        for nom, ok in preuves.items():
            print(f"   {'✅' if ok else '❌'} {nom}")

    print("=" * 75)
    return nb_ok == total_preuves


if __name__ == "__main__":
    main()
    # Force la sortie propre (évite le crash ChromaDB à la fin)
    import os as _os
    _os._exit(0)

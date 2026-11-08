from datetime import datetime, timedelta

import pytest

from nyx.agenda import Agenda, Evenement


@pytest.fixture
def ag(tmp_path):
    return Agenda(str(tmp_path / "agenda.json"))


def aujourdhui():
    return datetime.now().strftime("%Y-%m-%d")


def dans_n_jours(n):
    return (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d")


def test_ajouter_et_lister(ag):
    id1 = ag.ajouter("Dentiste", date=aujourdhui(), heure="14:00", rappel_min=30)
    id2 = ag.ajouter("Réunion", date=aujourdhui(), heure="18:00")
    tous = ag.tous()
    assert [e.id for e in tous] == [id1, id2]
    assert tous[0].titre == "Dentiste"
    assert tous[0].rappel_min == 30


def test_ajouter_valide_les_formats(ag):
    with pytest.raises(ValueError):
        ag.ajouter("")
    with pytest.raises(ValueError):
        ag.ajouter("x", date="pas une date")
    with pytest.raises(ValueError):
        ag.ajouter("x", date=aujourdhui(), heure="25:99")


def test_persistance(tmp_path):
    chemin = str(tmp_path / "agenda.json")
    a1 = Agenda(chemin)
    id1 = a1.ajouter("Dentiste", date=aujourdhui(), heure="14:00")
    a2 = Agenda(chemin)
    assert len(a2.tous()) == 1
    assert a2.tous()[0].id == id1


def test_du_jour_de_la_semaine_a_venir(ag):
    """Tests de filtrage sur trois horizons.

    NB : on utilise des dates RÉSOLUMENT FUTURES pour `de_la_semaine()` et `a_venir()`
    afin que le test passe quelle que soit l'heure d'exécution.
    """
    id_j = ag.ajouter("Aujourd'hui", date=aujourdhui(), heure="10:00")
    id_dem = ag.ajouter("Demain", date=dans_n_jours(1), heure="10:00")
    id_3j = ag.ajouter("Dans 3 jours", date=dans_n_jours(3), heure="10:00")
    id_loin = ag.ajouter("Dans 10 jours", date=dans_n_jours(10), heure="10:00")

    # du_jour : uniquement aujourd'hui (passé ou futur, peu importe)
    assert [e.id for e in ag.du_jour()] == [id_j]

    # de_la_semaine : aujourd'hui + 6 jours suivants → inclut aujourd'hui, demain, +3j, exclut +10j
    ids_semaine = [e.id for e in ag.de_la_semaine()]
    assert id_j in ids_semaine
    assert id_dem in ids_semaine
    assert id_3j in ids_semaine
    assert id_loin not in ids_semaine

    # a_venir : uniquement les événements FUTURS. Aujourd'hui à 10h peut être passé :
    # on vérifie donc seulement que les événements à venir bien futurs sont présents.
    ids_a_venir = [e.id for e in ag.a_venir()]
    assert id_dem in ids_a_venir
    assert id_3j in ids_a_venir
    assert id_loin in ids_a_venir


def test_supprimer_et_modifier(ag):
    id1 = ag.ajouter("x", date=aujourdhui(), heure="10:00")
    assert ag.supprimer("inconnu") is False
    assert ag.modifier(id1, titre="y", rappel_min=45) is True
    ev = ag.tous()[0]
    assert ev.titre == "y" and ev.rappel_min == 45
    assert ag.supprimer(id1) is True
    assert ag.tous() == []


def test_rappels_emis_une_seule_fois(ag):
    maintenant = datetime.now()
    dans_20min = (maintenant + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M")
    id1 = ag.ajouter("Rappel test", date=dans_20min[:10], heure=dans_20min[11:16], rappel_min=30)
    a_rappeler = ag.a_rappeler(maintenant)
    assert len(a_rappeler) == 1 and a_rappeler[0].id == id1
    ag.marquer_rappele(id1)
    assert ag.a_rappeler(maintenant) == []


def test_rappels_ignores_si_loin(ag):
    maintenant = datetime.now()
    dans_3h = (maintenant + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    ag.ajouter("Loin", date=dans_3h[:10], heure=dans_3h[11:16], rappel_min=15)
    assert ag.a_rappeler(maintenant) == []


def test_rappels_ignores_si_deja_passe(ag):
    maintenant = datetime.now()
    il_y_a_1h = (maintenant - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    ag.ajouter("Passé", date=il_y_a_1h[:10], heure=il_y_a_1h[11:16], rappel_min=15)
    assert ag.a_rappeler(maintenant) == []


def test_evenement_resume(ag):
    ev = Evenement(id="x", titre="Café", date="2026-10-15", heure="15:00",
                   duree_min=30, description="avec Xavier", rappel_min=10,
                   cree_le="2026-10-09 12:00:00")
    assert "15:00" in ev.resumer() and "avec Xavier" in ev.resumer()

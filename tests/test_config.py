import ast
import json
import math
import os

import pytest

from nyx.config import CHAMPS, ConfigNyx, ProxyConfig

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- non-régression avec l'ancien main.py

def _valeurs_de_l_ancien_main():
    """Extrait, par analyse du source, les valeurs par défaut de AgentCerebralNyx.__init__."""
    chemin = os.path.join(RACINE, "main.py")
    if not os.path.exists(chemin):
        pytest.skip("main.py absent : test de parité ignoré")
    with open(chemin, encoding="utf-8") as f:
        arbre = ast.parse(f.read())
    classe = next(n for n in arbre.body if isinstance(n, ast.ClassDef) and n.name == "AgentCerebralNyx")
    init = next(n for n in classe.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    valeurs = {}
    # paramètres du constructeur
    args = init.args
    for arg, defaut in zip(args.args[-len(args.defaults):], args.defaults):
        try:
            valeurs[arg.arg] = ast.literal_eval(defaut)
        except ValueError:
            pass
    # affectations self.X = <constante>
    for noeud in ast.walk(init):
        if isinstance(noeud, ast.Assign) and len(noeud.targets) == 1:
            cible = noeud.targets[0]
            if isinstance(cible, ast.Attribute) and isinstance(cible.value, ast.Name) and cible.value.id == "self":
                try:
                    valeurs[cible.attr] = ast.literal_eval(noeud.value)
                except ValueError:
                    pass
    return valeurs


def test_les_defauts_sont_identiques_a_l_ancien_main():
    ancien = _valeurs_de_l_ancien_main()
    defauts = ConfigNyx.defauts()
    compares = 0
    ecarts = []
    for nom in CHAMPS:
        if nom in ancien and nom not in ("model_consolidation",):  # consolidation : résolu à part
            compares += 1
            if getattr(defauts, nom) != ancien[nom]:
                ecarts.append(f"{nom}: config={getattr(defauts, nom)!r} ancien={ancien[nom]!r}")
    assert not ecarts, "défauts divergents : " + " ; ".join(ecarts)
    assert compares >= 28, f"seulement {compares} paramètres comparés : l'analyse du source a-t-elle échoué ?"


def test_tous_les_parametres_de_l_ancien_main_existent_dans_la_config():
    ancien = _valeurs_de_l_ancien_main()
    constantes = {n for n in ancien if n.isupper()}
    manquants = constantes - CHAMPS
    assert not manquants, f"constantes de l'ancien main absentes de la config : {sorted(manquants)}"


# ---------------------------------------------------------------- validation

def test_toutes_les_valeurs_par_defaut_sont_valides_et_coherentes():
    cfg = ConfigNyx.defauts()
    for nom in CHAMPS:
        assert ConfigNyx.valider(nom, getattr(cfg, nom)) == getattr(cfg, nom)
    assert cfg.coherence() == []


def test_bornes_et_types():
    cfg = ConfigNyx.defauts()
    with pytest.raises(ValueError):
        cfg.definir("SEUIL_DISTANCE", 99)           # hors bornes
    with pytest.raises(ValueError):
        cfg.definir("SEUIL_DISTANCE", "1.0")        # mauvais type
    with pytest.raises(ValueError):
        cfg.definir("SEUIL_DISTANCE", True)         # un booléen n'est pas un nombre
    with pytest.raises(ValueError):
        cfg.definir("SEUIL_DISTANCE", math.nan)
    with pytest.raises(ValueError):
        cfg.definir("sommeil_auto_actif", 1)        # booléen strict
    with pytest.raises(ValueError):
        cfg.definir("MAX_HISTORIQUE", 12.5)         # entier attendu
    with pytest.raises(ValueError):
        cfg.definir("nom", "   ")                   # texte non vide
    with pytest.raises(KeyError):
        cfg.definir("PARAMETRE_INEXISTANT", 1)
    assert cfg.SEUIL_DISTANCE == 1.2                # rien n'a été modifié par les échecs


def test_conversions_acceptees():
    cfg = ConfigNyx.defauts()
    assert cfg.definir("MAX_HISTORIQUE", 20.0) == 20 and isinstance(cfg.MAX_HISTORIQUE, int)
    assert cfg.definir("SEUIL_DISTANCE", 1) == 1.0 and isinstance(cfg.SEUIL_DISTANCE, float)
    assert cfg.definir("nom", "  Nyx2  ") == "Nyx2"


def test_creer_est_strict_et_resout_le_modele_de_consolidation():
    assert ConfigNyx.creer().model_consolidation == "qwen2.5:7b"
    assert ConfigNyx.creer(model_name="gemma2:2b").model_consolidation == "gemma2:2b"
    assert ConfigNyx.creer(model_name="a", model_consolidation="b").model_consolidation == "b"
    with pytest.raises(ValueError):
        ConfigNyx.creer(num_ctx=1)


def test_coherence_detecte_les_seuils_inverses():
    cfg = ConfigNyx.defauts()
    cfg.definir("SEUIL_ENVIE_DORMIR", 0.9)
    cfg.definir("SEUIL_OUBLI", 0.6)
    pb = cfg.coherence()
    assert len(pb) == 2 and any("SEUIL_ENVIE_DORMIR" in p for p in pb) and any("SEUIL_OUBLI" in p for p in pb)


def test_champs_decrits_sont_complets():
    decrits = ConfigNyx.champs_decrits()
    assert {d["nom"] for d in decrits} == CHAMPS
    for d in decrits:
        assert d["doc"] and d["groupe"], f"{d['nom']} sans description"
    numeriques = [d for d in decrits if d["type"] in ("int", "float")]
    assert all(d["mini"] is not None and d["maxi"] is not None for d in numeriques), "bornes manquantes"


# ---------------------------------------------------------------- fichier

def test_aller_retour_fichier(tmp_path):
    chemin = str(tmp_path / "cfg.json")
    cfg = ConfigNyx.defauts()
    cfg.definir("SEUIL_DISTANCE", 0.9)
    cfg.definir("model_name", "gemma2:2b")
    cfg.definir("sommeil_auto_actif", False)
    cfg.sauver(chemin)
    relu, avert = ConfigNyx.depuis_fichier(chemin)
    assert avert == []
    assert relu.SEUIL_DISTANCE == 0.9 and relu.model_name == "gemma2:2b" and relu.sommeil_auto_actif is False
    assert relu.en_dict() == cfg.en_dict()


def test_fichier_absent_donne_les_defauts(tmp_path):
    cfg, avert = ConfigNyx.depuis_fichier(str(tmp_path / "absent.json"))
    assert avert == [] and cfg.en_dict() == ConfigNyx.defauts().en_dict()


def test_fichier_corrompu_ne_plante_pas(tmp_path):
    chemin = tmp_path / "cfg.json"
    chemin.write_text("{ceci n'est pas du json", encoding="utf-8")
    cfg, avert = ConfigNyx.depuis_fichier(str(chemin))
    assert len(avert) == 1 and "illisible" in avert[0]
    assert cfg.en_dict() == ConfigNyx.defauts().en_dict()
    chemin.write_text("[1, 2, 3]", encoding="utf-8")
    assert "invalide" in ConfigNyx.depuis_fichier(str(chemin))[1][0]


def test_valeur_invalide_rejetee_sans_casser_le_reste(tmp_path):
    chemin = tmp_path / "cfg.json"
    chemin.write_text(json.dumps({"version": 1, "parametres": {
        "SEUIL_DISTANCE": 99,            # hors bornes
        "TAUX_DECAY": 0.1,               # valide
        "INCONNU": 5,                    # inconnu
        "debug": True,                   # non persistant
    }}), encoding="utf-8")
    cfg, avert = ConfigNyx.depuis_fichier(str(chemin))
    assert cfg.SEUIL_DISTANCE == 1.2 and cfg.TAUX_DECAY == 0.1 and cfg.debug is False
    texte = " | ".join(avert)
    assert "hors bornes" in texte and "INCONNU" in texte and "debug" in texte


def test_fichier_a_plat_et_version_future_toleres(tmp_path):
    chemin = tmp_path / "cfg.json"
    chemin.write_text(json.dumps({"version": 99, "TAUX_DECAY": 0.2}), encoding="utf-8")
    cfg, avert = ConfigNyx.depuis_fichier(str(chemin))
    assert cfg.TAUX_DECAY == 0.2 and any("plus récente" in a for a in avert)


def test_debug_n_est_jamais_sauvegarde(tmp_path):
    chemin = str(tmp_path / "cfg.json")
    cfg = ConfigNyx.defauts()
    cfg.definir("debug", True)
    cfg.sauver(chemin)
    with open(chemin, encoding="utf-8") as f:
        assert "debug" not in json.load(f)["parametres"]
    assert "debug" in cfg.en_dict(persistants_seulement=False)


def test_ecriture_atomique(tmp_path, monkeypatch):
    chemin = str(tmp_path / "cfg.json")
    ConfigNyx.defauts().sauver(chemin)
    avant = open(chemin, encoding="utf-8").read()
    assert not os.path.exists(chemin + ".tmp")

    def echec(*a, **k):
        raise OSError("disque plein simulé")
    monkeypatch.setattr(os, "replace", echec)
    cfg = ConfigNyx.defauts()
    cfg.definir("TAUX_DECAY", 0.5)
    with pytest.raises(OSError):
        cfg.sauver(chemin)
    assert open(chemin, encoding="utf-8").read() == avant, "l'ancien fichier doit rester intact"
    assert not os.path.exists(chemin + ".tmp"), "le fichier temporaire doit être nettoyé"


# ---------------------------------------------------------------- compatibilité (mixin)

class _FauxAgent(ProxyConfig):
    def __init__(self):
        self.cfg = ConfigNyx.defauts()
        self.autre = "valeur normale"


def test_proxy_lecture_ecriture_validation():
    a = _FauxAgent()
    assert a.SEUIL_DISTANCE == 1.2 and a.nom == "Nyx"
    a.SEUIL_DISTANCE = 0.8                      # comme le fait l'interface
    assert a.cfg.SEUIL_DISTANCE == 0.8 and a.SEUIL_DISTANCE == 0.8
    with pytest.raises(ValueError):
        a.SEUIL_DISTANCE = 99                   # la validation s'applique aussi par le proxy
    assert a.SEUIL_DISTANCE == 0.8
    a.debug = True
    assert a.cfg.debug is True
    a.autre = "x"                               # les autres attributs restent normaux
    assert a.autre == "x" and "autre" in a.__dict__
    with pytest.raises(AttributeError):
        _ = a.inexistant


def test_proxy_sans_cfg_ne_boucle_pas():
    class Brut(ProxyConfig):
        pass
    b = Brut()
    with pytest.raises(AttributeError):
        _ = b.SEUIL_DISTANCE
    b.valeur = 1
    assert b.valeur == 1

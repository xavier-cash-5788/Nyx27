"""
Tests unitaires du module TTS (nyx/tts.py).
Utilise FauxBackendTTS pour tester sans dépendances lourdes (transformers, torch, soundfile).
Couvre :
  - mapping émotion → style vocal
  - intensité émotionnelle (0.0, 0.5, 1.0)
  - streaming audio (callback sur_morceau)
  - gestion des erreurs (TTSIndisponible, VoixIntrouvable)
  - méthode verifier()
  - cas limites (texte vide, émotion inconnue)
"""
import pytest
from nyx.config import ConfigNyx
from nyx.tts import (
    ClientTTS,
    FauxBackendTTS,
    STYLE_EMOTION,
    TTSIndisponible,
    VoixIntrouvable,
    ErreurTTS,
)

# ------------------------------------------------------------------ fixtures
@pytest.fixture
def cfg_tts():
    """Config TTS par défaut."""
    cfg = ConfigNyx.defauts()
    cfg.definir("tts_actif", True)
    cfg.definir("model_tts", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    cfg.definir("voix_tts_speaker", "fr_female_1")
    cfg.definir("tts_use_gpu", False)
    cfg.definir("intensite_emotion", 0.7)
    return cfg

@pytest.fixture
def tts_faux(cfg_tts):
    """ClientTTS avec backend de test."""
    faux = FauxBackendTTS(duree_s=0.3, sampling_rate=24000)
    return ClientTTS(cfg_tts, backend=faux)

# ------------------------------------------------------------------ mapping émotion
def test_toutes_les_emotions_sont_mappees():
    """Les 11 émotions du limbique doivent avoir un style vocal."""
    emotions_limbique = [
        "Excitation joyeuse", "Surprise agréable", "Sérénité confiante",
        "Contentement calme", "Curiosité intense", "Intérêt modéré",
        "Colère", "Peur", "Frustration", "Tristesse", "Neutralité",
    ]
    for emotion in emotions_limbique:
        assert emotion in STYLE_EMOTION, f"Émotion {emotion!r} absente du mapping"
        assert isinstance(STYLE_EMOTION[emotion], str)
        assert len(STYLE_EMOTION[emotion]) > 0

def test_emotion_inconnue_utilise_neutralite(cfg_tts):
    """Une émotion inconnue doit utiliser le style neutre."""
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Émotion inexistante")
    assert len(faux.appels) == 1
    prompt = faux.appels[0]["prompt"]
    assert "neutral" in prompt.lower() or "calm" in prompt.lower()

# ------------------------------------------------------------------ synthèse basique
def test_parle_retourne_reponse_tts(tts_faux):
    """parler() doit retourner une ReponseTTS avec les bons champs."""
    rep = tts_faux.parler("Bonjour Xavier", emotion="Sérénité confiante")
    assert rep.duree_s > 0
    assert rep.modele == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert rep.voix == "fr_female_1"
    assert rep.emotion == "Sérénité confiante"
    assert rep.sampling_rate == 24000
    assert len(rep.audio_bytes) > 0

def test_parle_appelle_le_backend(tts_faux):
    """parler() doit appeler le backend avec le bon prompt et speaker."""
    tts_faux.parler("Test", emotion="Excitation joyeuse", intensite=0.8)
    assert len(tts_faux._backend.appels) == 1
    appel = tts_faux._backend.appels[0]
    assert "joyful" in appel["prompt"]
    assert appel["speaker"] == "fr_female_1"

def test_texte_vide_ne_genere_pas(tts_faux):
    """Un texte vide ou None ne doit pas appeler le backend."""
    rep = tts_faux.parler("", emotion="Neutralité")
    assert rep.audio_bytes == b""
    assert rep.duree_s == 0.0
    assert len(tts_faux._backend.appels) == 0

    rep = tts_faux.parler("   ", emotion="Neutralité")
    assert rep.audio_bytes == b""
    assert len(tts_faux._backend.appels) == 0

# ------------------------------------------------------------------ intensité émotionnelle
def test_intensite_faible_style_minimal(cfg_tts):
    """intensite=0.0 doit produire un style minimal (neutral, calm)."""
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Excitation joyeuse", intensite=0.0)
    prompt = faux.appels[0]["prompt"]
    assert "neutral" in prompt.lower()

def test_intensite_moyenne_style_partiel(cfg_tts):
    """intensite=0.5 doit garder les 3 premiers adjectifs."""
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Excitation joyeuse", intensite=0.5)
    prompt = faux.appels[0]["prompt"]
    # Le style complet est "joyful, energetic, enthusiastic, fast-paced, bright"
    # À 0.5, on garde les 3 premiers : "joyful, energetic, enthusiastic"
    assert "joyful" in prompt
    assert "energetic" in prompt
    assert "enthusiastic" in prompt
    assert "fast-paced" not in prompt  # 4e adjectif, exclu

def test_intensite_forte_style_complet(cfg_tts):
    """intensite=1.0 doit garder le style complet."""
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Excitation joyeuse", intensite=1.0)
    prompt = faux.appels[0]["prompt"]
    assert "joyful" in prompt
    assert "bright" in prompt  # 5e adjectif, inclus

def test_intensite_defaut_vient_de_la_config(cfg_tts):
    """Sans intensite explicite, cfg.intensite_emotion est utilisé."""
    cfg_tts.definir("intensite_emotion", 0.3)
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Colère")
    prompt = faux.appels[0]["prompt"]
    # À 0.3, style minimal
    assert "neutral" in prompt.lower()

def test_intensite_bornee_entre_0_et_1(cfg_tts):
    """intensite hors [0,1] doit être bornée."""
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    
    tts.parler("Test", emotion="Neutralité", intensite=-0.5)
    prompt_neg = faux.appels[0]["prompt"]
    
    tts.parler("Test", emotion="Neutralité", intensite=1.5)
    prompt_pos = faux.appels[1]["prompt"]
    
    # Les deux doivent produire un style (pas d'erreur)
    assert len(prompt_neg) > 0
    assert len(prompt_pos) > 0

# ------------------------------------------------------------------ streaming
def test_streaming_callback_appele(tts_faux):
    """Le callback sur_morceau doit être appelé avec les bytes audio."""
    morceaux = []
    rep = tts_faux.parler("Test", emotion="Neutralité", sur_morceau=morceaux.append)
    assert len(morceaux) == 1
    assert isinstance(morceaux[0], bytes)
    assert len(morceaux[0]) > 0
    assert morceaux[0] == rep.audio_bytes

def test_streaming_callback_erreur_ne_tue_pas(cfg_tts):
    """Une erreur dans sur_morceau ne doit pas tuer la synthèse."""
    def callback_bogue(_):
        raise RuntimeError("Bug simulé")
    
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    rep = tts.parler("Test", emotion="Neutralité", sur_morceau=callback_bogue)
    # La synthèse doit avoir réussi malgré le bug
    assert len(rep.audio_bytes) > 0

def test_sans_callback_streaming(tts_faux):
    """Sans callback, la synthèse doit fonctionner normalement."""
    rep = tts_faux.parler("Test", emotion="Neutralité")
    assert len(rep.audio_bytes) > 0

# ------------------------------------------------------------------ gestion des erreurs
def test_backend_indisponible_leve_tts_indisponible(cfg_tts):
    """Si le backend échoue au chargement, TTSIndisponible est levée."""
    faux = FauxBackendTTS(echec=True)
    tts = ClientTTS(cfg_tts, backend=faux)
    with pytest.raises(TTSIndisponible):
        tts.parler("Test", emotion="Neutralité")

def test_verifier_retourne_dict_sans_lever(cfg_tts):
    """verifier() doit retourner un dict sans lever, même en cas d'erreur."""
    # Cas OK
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    result = tts.verifier()
    assert result["disponible"] is True
    assert "prêt" in result["message"].lower()

    # Cas échec
    faux_echec = FauxBackendTTS(echec=True)
    tts_echec = ClientTTS(cfg_tts, backend=faux_echec)
    result = tts_echec.verifier()
    assert result["disponible"] is False
    assert "panne" in result["message"].lower()

def test_verifier_attrape_toutes_les_exceptions(cfg_tts):
    """verifier() doit attraper toutes les exceptions, même inattendues."""
    class BackendBogue:
        def charger(self):
            raise ValueError("Erreur inattendue")
        def synthetise(self, prompt, speaker=None):
            raise ValueError("Erreur inattendue")
    
    tts = ClientTTS(cfg_tts, backend=BackendBogue())
    result = tts.verifier()
    assert result["disponible"] is False
    assert "inattendue" in result["message"].lower()

# ------------------------------------------------------------------ statistiques
def test_dernier_appel_enregistre_stats(tts_faux):
    """Le dernier appel réussi doit être enregistré dans tts.dernier."""
    rep = tts_faux.parler("Test", emotion="Colère", intensite=0.9)
    assert tts_faux.dernier["modele"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert tts_faux.dernier["voix"] == "fr_female_1"
    assert tts_faux.dernier["emotion"] == "Colère"
    assert tts_faux.dernier["duree_s"] > 0
    assert tts_faux.dernier["taille_bytes"] > 0
    assert tts_faux.dernier["sampling_rate"] == 24000

# ------------------------------------------------------------------ cas limites
def test_texte_tres_long_ne_plante_pas(tts_faux):
    """Un texte très long ne doit pas faire planter la synthèse."""
    texte_long = "Mot " * 1000
    rep = tts_faux.parler(texte_long, emotion="Neutralité")
    assert len(rep.audio_bytes) > 0

def test_caracteres_speciaux_ne_plante_pas(tts_faux):
    """Des caractères spéciaux ne doivent pas faire planter la synthèse."""
    texte_special = "Test avec émojis 🎉 🌙 et accents àéîõü"
    rep = tts_faux.parler(texte_special, emotion="Neutralité")
    assert len(rep.audio_bytes) > 0

def test_speaker_vide_utilise_default(cfg_tts):
    """Un speaker vide doit utiliser la voix par défaut."""
    cfg_tts.definir("voix_tts_speaker", "")
    faux = FauxBackendTTS()
    tts = ClientTTS(cfg_tts, backend=faux)
    tts.parler("Test", emotion="Neutralité")
    assert faux.appels[0]["speaker"] is None

# ------------------------------------------------------------------ intégration avec ConfigNyx
def test_config_tts_persistante(tmp_path):
    """Les paramètres TTS doivent être sauvegardés et rechargés."""
    chemin = str(tmp_path / "cfg.json")
    cfg = ConfigNyx.defauts()
    cfg.definir("tts_actif", True)
    cfg.definir("model_tts", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    cfg.definir("voix_tts_speaker", "en_female_2")
    cfg.definir("intensite_emotion", 0.9)
    cfg.sauver(chemin)
    
    relu, avert = ConfigNyx.depuis_fichier(chemin)
    assert avert == []
    assert relu.tts_actif is True
    assert relu.model_tts == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert relu.voix_tts_speaker == "en_female_2"
    assert relu.intensite_emotion == 0.9

def test_config_tts_validation_bornes():
    """intensite_emotion doit être bornée entre 0 et 1."""
    cfg = ConfigNyx.defauts()
    with pytest.raises(ValueError):
        cfg.definir("intensite_emotion", -0.1)
    with pytest.raises(ValueError):
        cfg.definir("intensite_emotion", 1.1)
    
    cfg.definir("intensite_emotion", 0.5)
    assert cfg.intensite_emotion == 0.5
"""
nyx_gui.py : interface graphique moderne pour Nyx (customtkinter).

Installation :
    pip install customtkinter
    (Linux : sudo apt install python3-tk)

Lancement :
    placez ce fichier à côté du package nyx/, puis  python3 nyx_gui.py

Si les emojis s'affichent mal ou font planter Tk sur votre système :
    NYX_SANS_EMOJI=1 python3 nyx_gui.py

V21 :
  - Texte des bulles sélectionnable et copiable
  - Voix Piper uniquement
  - Rêves : 2 par sommeil (T dynamique 3 phases)
  - Démon de Dissonance : détection et résolution des conflits internes
  - Onglet 🧠 Dissonances
  - Onglet 🗜️ Archives compressées (.cpk, plafond 1 Go, purge auto)
  - Onglet 📚 Documents (lecture à la demande + lectures spontanées)
  - Envie de lire : jauge de pulsion + lectures spontanées
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
except ImportError:
    sys.exit("customtkinter est requis : pip install customtkinter")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from nyx.agent import AgentCerebralNyx  # noqa: E402


# ==================== STYLE ====================
BG = ("#f3f4f8", "#0e1016")
SURFACE = ("#ffffff", "#161923")
SURFACE2 = ("#e9ebf2", "#1f2331")
TEXT = ("#191c27", "#e7e9f3")
MUTED = ("#6a7086", "#8a90a6")
ACCENT, ACCENT_HOVER = "#7c5cff", "#6a49f0"
GOOD, WARN, BAD = "#2fbf71", "#f0a93b", "#ef5b5b"
OR = "#e8b84a"
VIOLET = "#a855f7"
VIOLET_FONCE = "#7c3aed"
INDIGO = "#6366f1"
INDIGO_FONCE = "#4f46e5"
TEAL = "#14b8a6"
TEAL_FONCE = "#0d9488"

EMOJIS_EMOTIONS = {
    "Excitation joyeuse": "🎉", "Surprise agréable": "😲", "Sérénité confiante": "😊",
    "Contentement calme": "😌", "Curiosité intense": "🧐", "Intérêt modéré": "🤔", "Colère": "😠",
    "Peur": "😨", "Frustration": "😤", "Tristesse": "😢", "Neutralité": "😐",
}

FICHIER_REGLAGES = "nyx_gui_settings.json"


def charger_reglages():
    try:
        with open(FICHIER_REGLAGES, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


REGLAGES = charger_reglages()
SANS_EMOJI = os.environ.get("NYX_SANS_EMOJI") == "1" or REGLAGES.get("sans_emoji", False)
_RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")


def T(texte):
    if not SANS_EMOJI:
        return texte
    return re.sub(r"\s{2,}", " ", _RE_EMOJI.sub("", texte)).strip()


def B(parent, texte, **kw):
    return ctk.CTkButton(parent, text=T(texte), **kw)


def L(parent, texte="", **kw):
    return ctk.CTkLabel(parent, text=T(texte), **kw)


def sombre():
    return ctk.get_appearance_mode() == "Dark"


def barre_texte(valeur, largeur=10):
    n = max(0, min(largeur, round(valeur * largeur)))
    return "█" * n + "░" * (largeur - n)


def couleur_taille(stats):
    if stats is None:
        return MUTED
    if stats.au_dessus_limite:
        return BAD
    if stats.au_dessus_avertissement:
        return WARN
    return GOOD


def emoji_taille(stats):
    if stats is None:
        return "⚪"
    if stats.au_dessus_limite:
        return "🔴"
    if stats.au_dessus_avertissement:
        return "🟡"
    return "🟢"


def couleur_archive(stats):
    if stats is None:
        return MUTED
    p = stats.get("pourcentage", 0.0)
    if p >= 100:
        return BAD
    if p >= 85:
        return WARN
    return GOOD


def emoji_archive(stats):
    if stats is None:
        return "⚪"
    p = stats.get("pourcentage", 0.0)
    if p >= 100:
        return "🔴"
    if p >= 85:
        return "🟡"
    return "🟢"


def couleur_fond(valeur):
    if isinstance(valeur, tuple):
        return valeur[1] if sombre() else valeur[0]
    return valeur


# ==================== CAPTURE CONSOLE ====================
class CaptureConsole:
    encoding = "utf-8"

    def __init__(self, file_ui, chemin):
        self.file_ui, self.chemin = file_ui, chemin
        self.lignes, self.base, self._partiel = [], 0, ""
        self.original = sys.__stdout__
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        self._f = open(chemin, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        if not s:
            return 0
        try:
            self.original.write(s)
        except Exception:
            pass
        self._partiel += s
        while "\n" in self._partiel:
            ligne, self._partiel = self._partiel.split("\n", 1)
            self.lignes.append(ligne)
            if len(self.lignes) > 6000:
                del self.lignes[:1000]
                self.base += 1000
            try:
                self._f.write(ligne + "\n")
            except Exception:
                pass
            self.file_ui.put(("log", ligne))
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def marque(self):
        return self.base + len(self.lignes)

    def depuis(self, marque):
        return self.lignes[max(0, marque - self.base):]

    def fermer(self):
        try:
            self._f.close()
        except Exception:
            pass


# ==================== APPLICATION ====================
class NyxApp(ctk.CTk):
    NAV = [("chat", "💬  Conversation"), ("memoire", "🧠  Mémoire"),
           ("journal", "📖  Journal intime"), ("reves", "🌙  Rêves"),
           ("dissonances", "🧠  Dissonances"),
           ("archives", "🗜️  Archives"),
           ("documents", "📚  Documents"),
           ("etat", "❤️  État  & émotions"), ("logs", "📜  Logs  & diagnostic"),
           ("outils", "🛠️  Outils"), ("reglages", "⚙️  Réglages")]
    FILTRES_MEM = [("Tous les souvenirs", "tous"), ("🔥 Marquants", "marquant"), ("🧱 Concepts", "concept"),
                   ("💭 Souvenirs bruts", "brut"), ("🌙 Oniriques", "onirique"),
                   ("💨 Faibles (à oublier)", "faible")]
    TRIS_MEM = ["Plus récents", "Plus anciens", "Plus forts", "Plus faibles", "Plus réactivés"]

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(REGLAGES.get("apparence", "dark"))
        ctk.set_default_color_theme("dark-blue")
        self.title("Nyx — Agent cérébral")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)
        self.agent = None
        self.occupe = False
        self.boutons_action = []
        self.donnees_memoire, self.donnees_journal = [], []
        self.donnees_archives = []
        self.donnees_documents = []
        self.console_visible = False
        self._toast_id = None
        self.stats_taille = None
        self.stats_archives_cache = None
        self.file_ui = queue.Queue()
        self.console = CaptureConsole(
            self.file_ui, os.path.join("logs", f"nyx_console_{datetime.now():%Y%m%d_%H%M%S}.txt"))
        sys.stdout = self.console
        self.overlay = None

        self._construire_entete()
        corps = ctk.CTkFrame(self, fg_color="transparent")
        corps.pack(fill="both", expand=True)
        self._construire_navigation(corps)
        self.zone = ctk.CTkFrame(corps, fg_color="transparent")
        self.zone.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=(12, 8))
        self._construire_barre_statut()
        self.pages = {
            "chat": self._page_chat(), "memoire": self._page_memoire(), "journal": self._page_journal(),
            "reves": self._page_reves(), "dissonances": self._page_dissonances(),
            "archives": self._page_archives(),
            "documents": self._page_documents(),
            "etat": self._page_etat(), "logs": self._page_logs(), "outils": self._page_outils(),
            "reglages": self._page_reglages(),
        }
        self._style_arbre()
        self.page_courante = None
        self.afficher_page("chat")
        self.protocol("WM_DELETE_WINDOW", self.quitter)

        # Raccourcis clavier globaux
        self.bind_all("<space>", self._raccourci_espace)
        self.bind_all("<Control-m>", lambda e: self.action_basculer_voix())
        self.bind_all("<Control-p>", lambda e: self.action_basculer_pensees())
        self.bind_all("<Control-r>", lambda e: self.action_rever())
        self.bind_all("<Control-d>", lambda e: self.action_afficher_dissonances())
        self.bind_all("<Control-a>", self._raccourci_archives)
        self.bind_all("<Control-l>", self._raccourci_lecture)

        self.after(80, self._poll)
        self._init_agent()

    # ---------- overlay ----------
    def _afficher_overlay(self, titre="Chargement de Nyx…", sous_titre=""):
        if self.overlay is not None:
            return
        self.overlay = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        centre = ctk.CTkFrame(self.overlay, fg_color="transparent")
        centre.place(relx=0.5, rely=0.5, anchor="center")
        L(centre, "🌙  Nyx", font=ctk.CTkFont(size=42, weight="bold"),
          text_color=ACCENT).pack(pady=(0, 20))
        L(centre, titre, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(0, 8))
        if sous_titre:
            L(centre, sous_titre, font=ctk.CTkFont(size=12), text_color=MUTED).pack(pady=(0, 20))
        self.prog_overlay = ctk.CTkProgressBar(centre, width=400, height=10, mode="indeterminate",
                                               progress_color=ACCENT)
        self.prog_overlay.pack(pady=10)
        self.prog_overlay.start()
        self.lbl_overlay_etape = L(centre, "Chargement de la mémoire…",
                                   font=ctk.CTkFont(size=11), text_color=MUTED)
        self.lbl_overlay_etape.pack(pady=(20, 0))

    def _maj_overlay(self, etape):
        if self.overlay is not None and hasattr(self, "lbl_overlay_etape"):
            self.lbl_overlay_etape.configure(text=etape)

    def _retirer_overlay(self):
        if self.overlay is not None:
            try:
                self.prog_overlay.stop()
            except Exception:
                pass
            self.overlay.destroy()
            self.overlay = None

    # ---------- exécution en arrière-plan ----------
    def _poll(self):
        try:
            while True:
                genre, contenu = self.file_ui.get_nowait()
                try:
                    if genre == "call":
                        contenu()
                    else:
                        self._sur_log(contenu)
                except Exception:
                    traceback.print_exc(file=sys.__stderr__)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def run_async(self, travail, apres=None, libelle="Nyx travaille…", exclusif=True, sur_erreur=None):
        if exclusif:
            if self.occupe:
                self.toast("⏳ Une opération est déjà en cours, patientez un instant.", WARN)
                return False
            self.occupe = True
            self._etat_occupe(True, libelle)
        marque = self.console.marque()

        def cible():
            resultat, erreur = None, None
            try:
                resultat = travail()
            except Exception as e:
                erreur = e
                print(traceback.format_exc())

            def fin():
                if exclusif:
                    self.occupe = False
                    self._etat_occupe(False)
                if erreur:
                    self.toast(f"❌ {type(erreur).__name__} : {erreur}", BAD)
                    if sur_erreur:
                        sur_erreur(erreur)
                elif apres:
                    apres(resultat, "\n".join(self.console.depuis(marque)).strip())

            self.file_ui.put(("call", fin))

        threading.Thread(target=cible, daemon=True).start()
        return True

    def _etat_occupe(self, occupe, libelle=""):
        etat = "disabled" if occupe else "normal"
        for b in self.boutons_action:
            b.configure(state=etat)
        if occupe:
            self.lbl_occupe.configure(text=T(f"⏳ {libelle}"))
            self.progression.pack(side="right", padx=(0, 16))
            self.progression.start()
            self._pastille(libelle, WARN)
        else:
            self.lbl_occupe.configure(text="")
            self.progression.stop()
            self.progression.pack_forget()
            self._pastille(self.agent.model_name if self.agent else "…", GOOD)

    def toast(self, texte, couleur=None):
        self.lbl_statut.configure(text=T(texte), text_color=couleur or MUTED)
        if self._toast_id:
            self.after_cancel(self._toast_id)
        self._toast_id = self.after(7000, lambda: self.lbl_statut.configure(text=""))

    # ---------- squelette ----------
    def _construire_entete(self):
        entete = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=SURFACE)
        entete.pack(fill="x")
        entete.pack_propagate(False)
        L(entete, "🌙  Nyx", font=ctk.CTkFont(size=21, weight="bold"), text_color=ACCENT).pack(
            side="left", padx=(20, 16))
        for titre, fabrique in (("Fichier", self._menu_fichier), ("Affichage", self._menu_affichage),
                                ("Actions", self._menu_actions), ("Voix", self._menu_voix),
                                ("Rêves", self._menu_reves), ("Dissonances", self._menu_dissonances),
                                ("Archives", self._menu_archives),
                                ("Documents", self._menu_documents),
                                ("Aide", self._menu_aide)):
            bouton = B(entete, f"{titre} ▾", width=90, height=32, fg_color="transparent",
                       hover_color=SURFACE2, text_color=TEXT)
            bouton.configure(command=lambda b=bouton, f=fabrique: self._ouvrir_menu(b, f))
            bouton.pack(side="left", padx=2)
        self.pastille = L(entete, "● Initialisation…", text_color=WARN)
        self.pastille.pack(side="right", padx=20)

    def _pastille(self, texte, couleur):
        self.pastille.configure(text=T(f"● {texte}"), text_color=couleur)

    def _construire_navigation(self, parent):
        nav = ctk.CTkFrame(parent, width=220, corner_radius=0, fg_color=SURFACE)
        nav.pack(side="left", fill="y", padx=(0, 16))
        nav.pack_propagate(False)
        L(nav, "NAVIGATION", text_color=MUTED, font=ctk.CTkFont(size=11, weight="bold")).pack(
            anchor="w", padx=22, pady=(20, 6))
        self.nav_boutons = {}
        for cle, libelle in self.NAV:
            b = B(nav, libelle, anchor="w", height=40, corner_radius=10, fg_color="transparent",
                  hover_color=SURFACE2, text_color=TEXT, command=lambda k=cle: self.afficher_page(k))
            b.pack(fill="x", padx=12, pady=2)
            self.nav_boutons[cle] = b

    def _construire_barre_statut(self):
        barre = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color=SURFACE)
        barre.pack(fill="x", side="bottom")
        self.lbl_statut = L(barre, "", text_color=MUTED, anchor="w")
        self.lbl_statut.pack(side="left", padx=16)
        self.lbl_voix_statut = L(barre, "🔊", text_color=GOOD)
        self.lbl_voix_statut.pack(side="right", padx=(0, 10))
        self.progression = ctk.CTkProgressBar(barre, mode="indeterminate", width=160, height=6,
                                              progress_color=ACCENT)
        self.lbl_occupe = L(barre, "", text_color=WARN)
        self.lbl_occupe.pack(side="right", padx=(0, 10))

    def afficher_page(self, cle):
        for k, page in self.pages.items():
            page.pack_forget()
            self.nav_boutons[k].configure(fg_color=ACCENT if k == cle else "transparent",
                                          text_color="#ffffff" if k == cle else TEXT)
        self.pages[cle].pack(fill="both", expand=True)
        self.page_courante = cle
        self.console_visible = cle == "logs" and self.selection_logs.get().startswith(T("📟"))
        self.rafraichir(cle)

    def rafraichir(self, cle=None):
        cle = cle or self.page_courante
        if not self.agent:
            return
        if cle == "memoire":
            self.charger_memoire()
        elif cle == "journal":
            self.charger_journal()
        elif cle == "reves":
            self.charger_reves()
        elif cle == "dissonances":
            self.charger_dissonances()
        elif cle == "archives":
            self.charger_archives()
        elif cle == "documents":
            self.charger_documents()
        elif cle == "etat":
            self.charger_etat()
        elif cle == "logs":
            self.charger_liste_logs()

    def _carte(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=14, **kw)

    def _option(self, parent, valeurs, commande=None, largeur=200):
        return ctk.CTkOptionMenu(parent, values=[T(v) for v in valeurs], command=commande, width=largeur,
                                 fg_color=SURFACE2, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                                 text_color=TEXT, dropdown_fg_color=SURFACE, dropdown_text_color=TEXT,
                                 dropdown_hover_color=SURFACE2, dynamic_resizing=False)

    # ---------- menus ----------
    def _nouveau_menu(self):
        s = sombre()
        return tk.Menu(self, tearoff=0, bg="#1f2331" if s else "#ffffff", fg="#e7e9f3" if s else "#191c27",
                       activebackground=ACCENT, activeforeground="#ffffff", bd=0, relief="flat")

    def _ouvrir_menu(self, bouton, fabrique):
        menu = fabrique()
        try:
            menu.tk_popup(bouton.winfo_rootx(), bouton.winfo_rooty() + bouton.winfo_height())
        finally:
            menu.grab_release()

    def _menu_fichier(self):
        m = self._nouveau_menu()
        m.add_command(label=T("📂  Ouvrir le dossier des logs"), command=lambda: self.ouvrir_dossier("logs"))
        m.add_command(label=T("📂  Ouvrir le dossier de la mémoire"),
                      command=lambda: self.ouvrir_dossier(self.agent.chemin_db if self.agent else "."))
        m.add_command(label=T("📂  Ouvrir le dossier des archives"),
                      command=lambda: self.ouvrir_dossier(
                          self.agent.cfg.archives_dossier if self.agent else "archives"))
        m.add_command(label=T("📂  Ouvrir le dossier des documents"),
                      command=lambda: self.ouvrir_dossier("documents"))
        m.add_separator()
        m.add_command(label=T("🔄  Tout actualiser"), command=self.rafraichir)
        m.add_separator()
        m.add_command(label=T("⏻  Quitter"), command=self.quitter)
        return m

    def _menu_affichage(self):
        m = self._nouveau_menu()
        for cle, libelle in self.NAV:
            m.add_command(label=T(libelle.strip()), command=lambda k=cle: self.afficher_page(k))
        m.add_separator()
        for mode, libelle in (("dark", "🌙  Thème sombre"), ("light", "☀️  Thème clair"),
                              ("system", "💻  Thème du système")):
            m.add_command(label=T(libelle), command=lambda md=mode: self.changer_apparence(md))
        return m

    def _menu_actions(self):
        m = self._nouveau_menu()
        m.add_command(label=T("💤  Dormir (NREM + 2 rêves REM)"), command=self.action_dormir)
        m.add_command(label=T("🪞  Identité (Nyx se décrit)"), command=self.action_identite)
        m.add_command(label=T("🧹  Nettoyer les concepts pollués"), command=self.action_nettoyer)
        m.add_separator()
        m.add_command(label=T("📖  Relire un document (spontané)"), command=self.action_lecture_spontanee)
        m.add_separator()
        m.add_command(label=T("💾  Purger la mémoire maintenant"), command=self.action_purger_memoire)
        m.add_command(label=T("📊  Vérifier la taille de la mémoire"), command=self.action_verifier_taille)
        m.add_separator()
        m.add_command(label=T("🔎  Check complet"), command=lambda: self.action_check(False))
        m.add_command(label=T("⚡  Check rapide"), command=lambda: self.action_check(True))
        m.add_separator()
        m.add_command(label=T("👍  Feedback positif"), command=lambda: self.feedback(1))
        m.add_command(label=T("👎  Feedback négatif"), command=lambda: self.feedback(-1))
        return m

    def _menu_voix(self):
        m = self._nouveau_menu()
        m.add_command(label=T("🔊  Basculer la voix (Ctrl+M)"), command=self.action_basculer_voix)
        m.add_command(label=T("💭  Basculer pensées à voix haute (Ctrl+P)"),
                      command=self.action_basculer_pensees)
        m.add_separator()
        m.add_command(label=T("▶️  Tester la voix"), command=self.action_tester_voix)
        m.add_command(label=T("⏹️  Arrêter la lecture (Espace)"), command=self.action_arreter_voix)
        m.add_separator()
        m.add_command(label=T("🔄  Recharger la voix (Piper)"), command=self.action_recharger_voix)
        m.add_command(label=T("📊  État de la voix"), command=self.action_etat_voix)
        return m

    def _menu_reves(self):
        m = self._nouveau_menu()
        m.add_command(label=T("🌙  Rêver maintenant (Ctrl+R)"), command=self.action_rever)
        m.add_command(label=T("📜  Afficher les derniers rêves"), command=self.action_afficher_reves)
        m.add_separator()
        m.add_command(label=T("📖  Ouvrir le journal (rêves)"),
                      command=lambda: self.afficher_page("journal"))
        m.add_separator()
        m.add_command(label=T("💤  Dormir (NREM + REM)"), command=self.action_dormir)
        return m

    def _menu_dissonances(self):
        m = self._nouveau_menu()
        m.add_command(label=T("🧠  Voir l'état (Ctrl+D)"), command=self.action_afficher_dissonances)
        m.add_command(label=T("🔄  Actualiser"), command=self.charger_dissonances)
        m.add_separator()
        m.add_command(label=T("🔀  Basculer simulation ↔ action"), command=self.action_basculer_simulation)
        m.add_command(label=T("⏸️  Mettre en pause"), command=self.action_pause_dissonance)
        m.add_command(label=T("▶️  Reprendre"), command=self.action_reprendre_dissonance)
        return m

    def _menu_archives(self):
        m = self._nouveau_menu()
        m.add_command(label=T("🗜️  Ouvrir l'onglet Archives (Ctrl+A)"),
                      command=lambda: self.afficher_page("archives"))
        m.add_command(label=T("➕  Nouvelle note"), command=self.action_archiver_note)
        m.add_command(label=T("🔄  Actualiser"), command=self.charger_archives)
        m.add_separator()
        m.add_command(label=T("📤  Exporter tout"), command=self.action_exporter_archives)
        m.add_command(label=T("📂  Ouvrir le dossier des archives"),
                      command=lambda: self.ouvrir_dossier(
                          self.agent.cfg.archives_dossier if self.agent else "archives"))
        return m

    def _menu_documents(self):
        m = self._nouveau_menu()
        m.add_command(label=T("📚  Ouvrir l'onglet Documents (Ctrl+L)"),
                      command=lambda: self.afficher_page("documents"))
        m.add_command(label=T("📖  Relire un document (spontané)"), command=self.action_lecture_spontanee)
        m.add_command(label=T("🔄  Actualiser"), command=self.charger_documents)
        m.add_separator()
        m.add_command(label=T("📂  Ouvrir le dossier des documents"),
                      command=lambda: self.ouvrir_dossier("documents"))
        return m

    def _menu_aide(self):
        m = self._nouveau_menu()
        m.add_command(label=T("📖  Commandes et fonctions"), command=self.afficher_aide)
        m.add_command(label=T("ℹ️  À propos"), command=lambda: messagebox.showinfo(
            "À propos", "Nyx — agent à mémoire persistante, émotions, rêves, journal intime, "
                        "documents et archives.\n"
                        "Interface : customtkinter. Toute la logique est dans le package nyx/.\n"
                        "Embeddings : nomic-embed-text via Ollama.\n"
                        "Voix : Piper TTS (Siwis FR).\n"
                        "Rêves : 2 par sommeil (T dynamique 3 phases).\n"
                        "Démon de Dissonance : détecte et résout les conflits internes.\n"
                        "Archives : notes compressées .cpk (plafond 1 Go).\n"
                        "Documents : écriture + lecture (à la demande ou spontanée)."))
        return m

    # ---------- page : conversation ----------
    def _page_chat(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)
        gauche = ctk.CTkFrame(page, fg_color="transparent")
        gauche.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        gauche.grid_columnconfigure(0, weight=1)
        gauche.grid_rowconfigure(0, weight=1)
        self.fil = ctk.CTkScrollableFrame(gauche, fg_color=SURFACE, corner_radius=14)
        self.fil.grid(row=0, column=0, sticky="nsew")
        barre = self._carte(gauche)
        barre.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        barre.grid_columnconfigure(0, weight=1)
        self.saisie = ctk.CTkTextbox(barre, height=66, fg_color=SURFACE2, corner_radius=10, wrap="word")
        self.saisie.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        self.saisie.bind("<Return>", self._entree_envoyer)
        boutons = ctk.CTkFrame(barre, fg_color="transparent")
        boutons.grid(row=0, column=1, padx=(0, 12), pady=12)
        envoyer = B(boutons, "Envoyer  ➤", width=120, height=32, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                    command=self.envoyer)
        envoyer.grid(row=0, column=0, columnspan=2, pady=(0, 6))
        pouce_haut = B(boutons, "👍", width=56, height=28, fg_color=SURFACE2, hover_color=GOOD,
                       text_color=TEXT, command=lambda: self.feedback(1))
        pouce_bas = B(boutons, "👎", width=56, height=28, fg_color=SURFACE2, hover_color=BAD,
                      text_color=TEXT, command=lambda: self.feedback(-1))
        pouce_haut.grid(row=1, column=0, padx=(0, 4))
        pouce_bas.grid(row=1, column=1)
        self.boutons_action += [envoyer, pouce_haut, pouce_bas]

        droite = ctk.CTkScrollableFrame(page, width=270, fg_color=SURFACE, corner_radius=14)
        droite.grid(row=0, column=1, sticky="ns")

        # Bouton MUTE
        self.btn_voix = B(droite, "🔊  Voix : ON", height=40,
                          fg_color=GOOD, hover_color="#26a862", text_color="#ffffff",
                          font=ctk.CTkFont(size=14, weight="bold"),
                          command=self.action_basculer_voix)
        self.btn_voix.pack(fill="x", padx=10, pady=(12, 4))
        self.btn_pensees = B(droite, "💭  Pensées : ON", height=28,
                             fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
                             command=self.action_basculer_pensees)
        self.btn_pensees.pack(fill="x", padx=10, pady=(0, 6))

        # Bouton RÊVER
        btn_rever = B(droite, "🌙  Rêver maintenant (Ctrl+R)", height=36,
                      fg_color=VIOLET, hover_color=VIOLET_FONCE, text_color="#ffffff",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self.action_rever)
        btn_rever.pack(fill="x", padx=10, pady=(0, 6))
        self.boutons_action.append(btn_rever)

        # Bouton DISSONANCES
        btn_diss = B(droite, "🧠  Dissonances (Ctrl+D)", height=32,
                     fg_color=INDIGO, hover_color=INDIGO_FONCE, text_color="#ffffff",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     command=self.action_afficher_dissonances)
        btn_diss.pack(fill="x", padx=10, pady=(0, 6))
        self.boutons_action.append(btn_diss)

        # Bouton ARCHIVES
        btn_archives = B(droite, "🗜️  Archives (Ctrl+A)", height=32,
                         fg_color=TEAL, hover_color=TEAL_FONCE, text_color="#ffffff",
                         font=ctk.CTkFont(size=13, weight="bold"),
                         command=lambda: self.afficher_page("archives"))
        btn_archives.pack(fill="x", padx=10, pady=(0, 6))
        self.boutons_action.append(btn_archives)

        # Bouton DOCUMENTS
        btn_docs = B(droite, "📚  Documents (Ctrl+L)", height=32,
                     fg_color=TEAL, hover_color=TEAL_FONCE, text_color="#ffffff",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     command=lambda: self.afficher_page("documents"))
        btn_docs.pack(fill="x", padx=10, pady=(0, 6))
        self.boutons_action.append(btn_docs)

        B(droite, "⏹️  Arrêter la voix (Espace)", height=26, fg_color=SURFACE2,
          hover_color=BAD, text_color=TEXT,
          command=self.action_arreter_voix).pack(fill="x", padx=10, pady=(0, 10))

        L(droite, "État émotionnel", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 0))
        self.lbl_emoji = L(droite, "😐", font=ctk.CTkFont(size=42))
        self.lbl_emoji.pack(pady=(2, 0))
        self.lbl_emotion = L(droite, "Neutralité", text_color=ACCENT, font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_emotion.pack(pady=(0, 8))
        self.jauges_chat = {}
        for cle, nom in (("valence", "Valence"), ("activation", "Activation"), ("dominance", "Dominance")):
            ligne = ctk.CTkFrame(droite, fg_color="transparent")
            ligne.pack(fill="x", padx=18, pady=5)
            L(ligne, nom, text_color=MUTED, anchor="w").pack(side="left")
            valeur = L(ligne, "0.00", anchor="e")
            valeur.pack(side="right")
            barre_p = ctk.CTkProgressBar(droite, height=8, progress_color=ACCENT)
            barre_p.set(0.5)
            barre_p.pack(fill="x", padx=18, pady=(0, 6))
            self.jauges_chat[cle] = (barre_p, valeur)

        L(droite, "Pulsions", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(14, 2))
        self.jauges_pulsions_chat = {}
        for cle, nom in (("fatigue", "😴 Fatigue"),
                         ("envie_reflexion", "💭 Envie de penser"),
                         ("envie_lire", "📖 Envie de lire")):
            ligne = ctk.CTkFrame(droite, fg_color="transparent")
            ligne.pack(fill="x", padx=14, pady=4)
            L(ligne, nom, text_color=MUTED, anchor="w").pack(side="left")
            valeur = L(ligne, "0%", anchor="e")
            valeur.pack(side="right")
            barre_p = ctk.CTkProgressBar(droite, height=8, progress_color=ACCENT)
            barre_p.set(0)
            barre_p.pack(fill="x", padx=14, pady=(0, 4))
            self.jauges_pulsions_chat[cle] = (barre_p, valeur)

        L(droite, "Mémoire disque", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(14, 2))
        self.lbl_taille_chat = L(droite, "…", text_color=MUTED, justify="left", anchor="w",
                                 font=ctk.CTkFont(size=11))
        self.lbl_taille_chat.pack(fill="x", padx=14, pady=(0, 4))
        self.barre_taille_chat = ctk.CTkProgressBar(droite, height=8, progress_color=GOOD)
        self.barre_taille_chat.set(0)
        self.barre_taille_chat.pack(fill="x", padx=14, pady=(0, 4))

        L(droite, "Archives", font=ctk.CTkFont(size=14, weight="bold"),
          text_color=TEAL).pack(pady=(10, 2))
        self.lbl_archives_chat = L(droite, "…", text_color=MUTED, justify="left", anchor="w",
                                   font=ctk.CTkFont(size=11))
        self.lbl_archives_chat.pack(fill="x", padx=14, pady=(0, 4))
        self.barre_archives_chat = ctk.CTkProgressBar(droite, height=8, progress_color=TEAL)
        self.barre_archives_chat.set(0)
        self.barre_archives_chat.pack(fill="x", padx=14, pady=(0, 4))

        L(droite, "Dernier échange", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(14, 2))
        self.lbl_pipeline = L(droite, "Signal → Filtre émotionnel →\nDécodage mnésique → Décision →\nComportement",
                              text_color=MUTED, justify="left", anchor="w", font=ctk.CTkFont(size=11))
        self.lbl_pipeline.pack(padx=14, anchor="w")
        self.lbl_stats_chat = L(droite, "", text_color=MUTED, justify="left")
        self.lbl_stats_chat.pack(pady=(12, 6), padx=14, anchor="w")
        self.lbl_embedding_chat = L(droite, "", text_color=MUTED, justify="left", anchor="w",
                                    font=ctk.CTkFont(size=10))
        self.lbl_embedding_chat.pack(padx=14, anchor="w", pady=(0, 4))
        self.lbl_voix_chat = L(droite, "", text_color=MUTED, justify="left", anchor="w",
                               font=ctk.CTkFont(size=10))
        self.lbl_voix_chat.pack(padx=14, anchor="w", pady=(0, 8))

        for texte, cmd in (("💤  Dormir", self.action_dormir), ("💭  Penser à voix haute", self.action_reflechir),
                           ("📖  Lire un document", self.action_lecture_spontanee),
                           ("🪞  Identité", self.action_identite), ("💾  Purger mémoire", self.action_purger_memoire)):
            b = B(droite, texte, height=32, fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT, command=cmd)
            b.pack(fill="x", padx=14, pady=3)
            self.boutons_action.append(b)
        return page

    def _entree_envoyer(self, event):
        if event.state & 0x0001:
            return None
        self.envoyer()
        return "break"

    LEGENDES_SPONTANEES = {
        "pensee": "💭 Nyx · pense à voix haute",
        "sommeil": "😴 Nyx · a sommeil",
        "reveil": "🌅 Nyx · réveil",
        "interpellation": "📢 Nyx · t'interpelle",
        "reve": "🌙 Nyx · rêve",
        "dissonance": "🧠 Nyx · dissonance",
        "archive": "🗜️ Nyx · archive",
        "lecture_spontanee": "📖 Nyx · relit un document",
    }

    @staticmethod
    def _estimer_hauteur_texte(texte, largeur=560, font_size=13):
        if not texte:
            return 30
        chars_par_ligne = max(20, int(largeur / (font_size * 0.6)))
        lignes = 0
        for paragraphe in texte.split("\n"):
            lignes += max(1, (len(paragraphe) // chars_par_ligne) + 1)
        hauteur = lignes * 20 + 30
        return min(max(hauteur, 40), 600)

    def ajouter_bulle(self, auteur, texte, erreur=False, attente=False, genre=None):
        rangee = ctk.CTkFrame(self.fil, fg_color="transparent")
        rangee.pack(fill="x", pady=5, padx=8)
        moi = auteur == "user"
        spontane = genre in self.LEGENDES_SPONTANEES
        avert_taille = genre == "avertissement_taille"
        interpellation = genre == "interpellation"
        reve = genre == "reve"
        dissonance = genre == "dissonance"
        archive = genre == "archive"
        lecture = genre == "lecture_spontanee"

        if erreur:
            fond = BAD
        elif avert_taille:
            fond = WARN
        elif interpellation:
            fond = OR
        elif reve:
            fond = VIOLET
        elif dissonance:
            fond = INDIGO
        elif archive:
            fond = TEAL
        elif lecture:
            fond = ACCENT
        elif moi:
            fond = ACCENT
        else:
            fond = SURFACE2
        couleur = "#ffffff" if (moi or erreur or avert_taille or interpellation
                                or reve or dissonance or archive or lecture) else TEXT
        couleur_entete = (TEAL_FONCE if archive else
                          (VIOLET_FONCE if reve else
                           (INDIGO_FONCE if dissonance else
                            (ACCENT if (spontane or lecture) else MUTED)))) \
            if not (moi or erreur or avert_taille or interpellation
                    or reve or dissonance or archive or lecture) else couleur

        bulle = ctk.CTkFrame(rangee, fg_color=fond, corner_radius=16,
                             border_width=1 if (spontane or avert_taille or interpellation
                                                or reve or dissonance or archive or lecture) else 0,
                             border_color=WARN if avert_taille else (
                                 OR if interpellation else (
                                     VIOLET_FONCE if reve else (
                                         INDIGO_FONCE if dissonance else (
                                             TEAL_FONCE if archive else ACCENT)))))
        bulle.pack(side="right" if moi else "left", padx=4)

        nom = (self.agent.utilisateur if self.agent else "Vous") if moi else "Nyx"
        if avert_taille:
            legende = "⚠️ Nyx · avertissement mémoire"
        elif spontane:
            legende = self.LEGENDES_SPONTANEES[genre]
        else:
            legende = nom
        L(bulle, f"{legende}  ·  {datetime.now():%H:%M}",
          text_color=couleur_entete,
          font=ctk.CTkFont(size=10), anchor="w").pack(anchor="w", padx=14, pady=(8, 0))

        hauteur = self._estimer_hauteur_texte(texte, largeur=560, font_size=13)
        boite = ctk.CTkTextbox(
            bulle,
            height=hauteur,
            fg_color="transparent",
            border_width=0,
            wrap="word",
            text_color=couleur,
            font=ctk.CTkFont(size=13,
                             slant="italic" if (attente or spontane or reve or dissonance or archive or lecture) else "roman"),
            activate_scrollbars=True,
            scrollbar_button_color=couleur_fond(fond),
            scrollbar_button_hover_color=couleur_fond(fond),
        )
        boite.pack(fill="x", padx=14, pady=(2, 10))
        boite.insert("1.0", texte)
        boite.configure(state="disabled")

        try:
            boite._textbox.configure(
                background=couleur_fond(fond),
                borderwidth=0,
                highlightthickness=0,
                selectbackground=ACCENT,
                selectforeground="#ffffff",
                inactiveselectbackground=ACCENT,
                padx=0,
                pady=0,
            )
        except Exception:
            pass

        boite.bind("<Button-3>", lambda e, t=texte: self._copier_menu(e, t))

        def _select_all(event):
            try:
                boite._textbox.tag_add("sel", "1.0", "end-1c")
                boite._textbox.mark_set("insert", "1.0")
            except Exception:
                pass
            return "break"
        boite.bind("<Control-a>", _select_all)

        self.after(60, lambda: self.fil._parent_canvas.yview_moveto(1.0))
        return rangee

    def _copier_menu(self, event, texte):
        m = self._nouveau_menu()
        m.add_command(label=T("📋  Copier le message entier"), command=lambda: self._copier(texte))
        m.add_command(label=T("📋  Copier la sélection"), command=self._copier_selection)
        m.tk_popup(event.x_root, event.y_root)
        m.grab_release()

    def _copier_selection(self):
        try:
            widget = self.focus_get()
            if widget is not None and hasattr(widget, "selection_get"):
                selection = widget.selection_get()
                if selection:
                    self._copier(selection)
                    return
        except Exception:
            pass
        self.toast("⚠️ Aucune sélection.", WARN)

    def _copier(self, texte):
        if not texte:
            self.toast("⚠️ Rien à copier.", WARN)
            return
        self.clipboard_clear()
        self.clipboard_append(texte)
        self.toast("📋 Copié dans le presse-papiers.", GOOD)

    def envoyer(self):
        if not self.agent:
            self.toast("Nyx s'initialise encore…", WARN)
            return
        texte = self.saisie.get("1.0", "end").strip()
        if not texte:
            return
        if self.occupe:
            self.toast("⏳ Nyx est occupée, patientez un instant.", WARN)
            return
        try:
            self.agent.arreter_voix()
        except Exception:
            pass
        self.saisie.delete("1.0", "end")
        self.ajouter_bulle("user", texte)
        attente = self.ajouter_bulle("nyx", "Nyx réfléchit…", attente=True)

        def apres(reponse, _sortie):
            attente.destroy()
            self.ajouter_bulle("nyx", reponse, erreur=reponse.startswith("[Erreur"))
            self.maj_emotions()
            self.rafraichir()

        self.run_async(lambda: self.agent.percevoir_et_repondre(texte), apres, "Nyx réfléchit…",
                       sur_erreur=lambda e: attente.destroy())

    def feedback(self, valeur):
        if not self.agent:
            return
        self.run_async(lambda: self.agent.donner_feedback(valeur),
                       lambda r, s: (self.toast("👍 Feedback positif enregistré." if valeur > 0
                                                else "👎 Feedback négatif enregistré.", GOOD),
                                     self.maj_emotions(), self.rafraichir()),
                       "Enregistrement du feedback…")

    # ---------- voix ----------
    def action_basculer_voix(self):
        if not self.agent:
            return
        etat = self.agent.basculer_voix()
        REGLAGES["voix_actif"] = etat
        self.sauver_reglages()
        self._maj_bouton_voix()
        if etat:
            self.toast("🔊 Voix activée.", GOOD)
            try:
                self.agent.dire("Voix activée.", bloquant=False, force=True)
            except Exception:
                pass
        else:
            self.toast("🔇 Voix coupée.", WARN)

    def action_basculer_pensees(self):
        if not self.agent:
            return
        etat = self.agent.basculer_pensees()
        REGLAGES["voix_pensees"] = etat
        self.sauver_reglages()
        self._maj_bouton_pensees()
        self.toast(f"💭 Pensées à voix haute {'activées' if etat else 'désactivées'}.",
                   GOOD if etat else WARN)

    def action_arreter_voix(self):
        if self.agent:
            self.agent.arreter_voix()
            self.toast("⏹️ Lecture arrêtée.", WARN)

    def action_tester_voix(self):
        if not self.agent or self.agent.voix is None:
            self.toast("Voix indisponible.", WARN)
            return
        info = self.agent.verifier_voix()
        if not info.disponible:
            self.toast(f"⚠️ {info.message}", WARN)
            return
        self.run_async(
            lambda: self.agent.dire("Bonjour, je suis Nyx. Ma voix fonctionne.",
                                    bloquant=True, force=True),
            lambda r, s: self.toast("🔊 Test de voix terminé.", GOOD),
            "Test de voix…")

    def action_etat_voix(self):
        if not self.agent:
            return
        info = self.agent.verifier_voix()
        cfg = self.agent.cfg
        lignes = [
            "🗣️ ÉTAT DE LA VOIX DE NYX",
            "",
            "Moteur          : Piper TTS (Siwis FR)",
            f"Disponible      : {'✅' if info.disponible else '❌'}",
            f"Modèle          : {info.modele}",
            f"Détails         : {info.chemin}",
            f"Activation      : {'🔊 ON' if cfg.voix_actif else '🔇 OFF'}",
            f"Pensées parlées : {'✅' if cfg.voix_pensees else '❌'}",
            f"Interpellation  : {'✅' if cfg.voix_interpeller else '❌'}",
            "",
            f"Message : {info.message}",
        ]
        self.fenetre_texte("🗣️ État de la voix", "\n".join(lignes))

    def action_recharger_voix(self):
        if not self.agent:
            return
        if not messagebox.askyesno(
                "Recharger la voix",
                "Recharger la voix Piper avec les paramètres actuels ?"):
            return

        def apres(r, s):
            self._maj_bouton_voix()
            self.maj_emotions()
            self.toast("🔄 Voix rechargée.", GOOD)

        self.run_async(self.agent.recharger_voix, apres, "Rechargement de la voix…")

    def _raccourci_espace(self, event):
        if isinstance(event.widget, (tk.Text, tk.Entry)) or \
                event.widget.winfo_class() in ("Text", "Entry"):
            return None
        self.action_arreter_voix()
        return "break"

    def _raccourci_archives(self, event=None):
        if event is not None:
            try:
                if isinstance(event.widget, (tk.Text, tk.Entry)) or \
                        event.widget.winfo_class() in ("Text", "Entry"):
                    return None
            except Exception:
                pass
        self.afficher_page("archives")
        return "break"

    def _raccourci_lecture(self, event=None):
        if event is not None:
            try:
                if isinstance(event.widget, (tk.Text, tk.Entry)) or \
                        event.widget.winfo_class() in ("Text", "Entry"):
                    return None
            except Exception:
                pass
        self.afficher_page("documents")
        return "break"

    def _maj_bouton_voix(self):
        if not hasattr(self, "btn_voix") or not self.agent:
            return
        if self.agent.cfg.voix_actif:
            self.btn_voix.configure(text=T("🔊  Voix : ON"), fg_color=GOOD)
            if hasattr(self, "lbl_voix_statut"):
                self.lbl_voix_statut.configure(text="🔊", text_color=GOOD)
        else:
            self.btn_voix.configure(text=T("🔇  Voix : MUTE"), fg_color=BAD)
            if hasattr(self, "lbl_voix_statut"):
                self.lbl_voix_statut.configure(text="🔇", text_color=BAD)

    def _maj_bouton_pensees(self):
        if not hasattr(self, "btn_pensees") or not self.agent:
            return
        if self.agent.cfg.voix_pensees:
            self.btn_pensees.configure(text=T("💭  Pensées : ON"), fg_color=ACCENT)
        else:
            self.btn_pensees.configure(text=T("💭  Pensées : OFF"), fg_color=SURFACE2)

    # ---------- rêves ----------
    def action_rever(self):
        if not self.agent:
            return
        if self.occupe:
            self.toast("⏳ Une opération est déjà en cours.", WARN)
            return
        if not self.agent.cfg.reve_actif:
            self.toast("⚠️ Les rêves sont désactivés dans les réglages.", WARN)
            return

        self.ajouter_bulle("nyx", "Je ferme les yeux et je m'endors…", attente=True)

        def apres(reves, sortie):
            if reves:
                for reve in reves:
                    self.ajouter_bulle(
                        "nyx",
                        f"[Rêve {reve.numero}/2 · T={reve.temperature:.1f}]\n{reve.texte}",
                        genre="reve"
                    )
                self.toast(f"🌙 {len(reves)} rêve(s) fait(s).", VIOLET)
            else:
                self.ajouter_bulle("nyx", "Je n'ai pas réussi à rêver (pas assez de souvenirs ?).",
                                   erreur=True)
            self.maj_emotions()
            self.rafraichir()

        self.run_async(self.agent.rever, apres, "Nyx rêve…")

    def action_afficher_reves(self):
        if not self.agent:
            return
        reves = self.agent.raconter_tous_derniers_reves()
        if not reves:
            self.fenetre_texte("🌙 Derniers rêves", "Aucun rêve récent.")
            return
        lignes = []
        for reve in reves:
            lignes.append(f"═══════════════════════════════════════")
            lignes.append(f"🌙 RÊVE {reve.numero} — T={reve.temperature:.1f}")
            lignes.append(f"   {reve.timestamp}")
            lignes.append(f"═══════════════════════════════════════")
            lignes.append("")
            lignes.append(reve.texte)
            lignes.append("")
        self.fenetre_texte("🌙 Derniers rêves", "\n".join(lignes))

    # ---------- page : rêves ----------
    def _page_reves(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        L(barre, "🌙  Rêves de Nyx", font=ctk.CTkFont(size=16, weight="bold"),
          text_color=VIOLET).pack(side="left", padx=20, pady=14)
        b_rever = B(barre, "🌙  Rêver maintenant (Ctrl+R)", fg_color=VIOLET,
                    hover_color=VIOLET_FONCE, text_color="#ffffff",
                    command=self.action_rever)
        b_rever.pack(side="right", padx=(6, 14), pady=12)
        self.boutons_action.append(b_rever)
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_reves).pack(side="right", padx=6)

        self.liste_reves = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.liste_reves.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        L(self.liste_reves, "Cliquez sur « Rêver maintenant » ou laissez Nyx dormir.",
          text_color=MUTED).pack(pady=40)
        return page

    def charger_reves(self):
        if not self.agent:
            return

        def lire():
            entrees = self.agent._entrees_journal_triees()
            reves = []
            for epoch, texte, meta in entrees:
                if (meta or {}).get("type_evenement") == "reve":
                    reves.append({"epoch": epoch, "texte": texte, "meta": meta or {}})
            return reves

        self.run_async(lire, self._apres_reves, exclusif=False)

    def _apres_reves(self, reves, _sortie):
        for enfant in self.liste_reves.winfo_children():
            enfant.destroy()
        if not reves:
            L(self.liste_reves, "Aucun rêve enregistré. Cliquez sur « Rêver maintenant ».",
              text_color=MUTED).pack(pady=40)
            return

        reves.sort(key=lambda e: e["epoch"], reverse=True)

        for reve in reves[:50]:
            meta = reve["meta"]
            carte = ctk.CTkFrame(self.liste_reves, fg_color=SURFACE, corner_radius=14,
                                 border_width=2, border_color=VIOLET_FONCE)
            carte.pack(fill="x", pady=6, padx=2)

            tete = ctk.CTkFrame(carte, fg_color="transparent")
            tete.pack(fill="x", padx=16, pady=(12, 2))
            L(tete, "🌙  RÊVE", text_color=VIOLET, font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            L(tete, meta.get("timestamp", "?"), text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(side="right")

            hauteur = self._estimer_hauteur_texte(reve["texte"], largeur=900, font_size=13)
            boite = ctk.CTkTextbox(carte, height=hauteur, fg_color="transparent",
                                   border_width=0, wrap="word", text_color=TEXT,
                                   font=ctk.CTkFont(size=13), activate_scrollbars=True)
            boite.pack(fill="x", padx=16, pady=(2, 14))
            boite.insert("1.0", reve["texte"])
            boite.configure(state="disabled")
            try:
                boite._textbox.configure(background=couleur_fond(SURFACE), borderwidth=0,
                                         highlightthickness=0, selectbackground=ACCENT,
                                         selectforeground="#ffffff")
            except Exception:
                pass

    # ---------- page : dissonances ----------
    def _page_dissonances(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        L(barre, "🧠  Démon de Dissonance", font=ctk.CTkFont(size=16, weight="bold"),
          text_color=INDIGO).pack(side="left", padx=20, pady=14)
        b_toggle = B(barre, "🔀  Simulation ↔ Action", fg_color=INDIGO,
                     hover_color=INDIGO_FONCE, text_color="#ffffff",
                     command=self.action_basculer_simulation)
        b_toggle.pack(side="right", padx=(6, 14), pady=12)
        self.boutons_action.append(b_toggle)
        B(barre, "⏸️  Pause", fg_color=SURFACE2, hover_color=WARN, text_color=TEXT,
          command=self.action_pause_dissonance).pack(side="right", padx=6)
        B(barre, "▶️  Reprendre", fg_color=SURFACE2, hover_color=GOOD, text_color=TEXT,
          command=self.action_reprendre_dissonance).pack(side="right", padx=6)
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_dissonances).pack(side="right", padx=6)

        self.lbl_dissonance_diag = L(page, "…", text_color=MUTED, anchor="w",
                                     font=ctk.CTkFont(size=12), justify="left")
        self.lbl_dissonance_diag.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 10))

        self.liste_dissonances = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.liste_dissonances.grid(row=2, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)

        L(self.liste_dissonances, "Aucune résolution enregistrée.",
          text_color=MUTED).pack(pady=40)
        return page

    def charger_dissonances(self):
        if not self.agent or self.agent.demon is None:
            return

        def lire():
            diag = self.agent.demon.diagnostique()
            historique = list(self.agent.historique_dissonances(30))
            return {"diag": diag, "historique": historique}

        self.run_async(lire, self._apres_dissonances, exclusif=False)

    def _apres_dissonances(self, data, _sortie):
        if data is None:
            return
        self.lbl_dissonance_diag.configure(text=data["diag"])
        for enfant in self.liste_dissonances.winfo_children():
            enfant.destroy()
        historique = data["historique"]
        if not historique:
            L(self.liste_dissonances, "Aucune résolution enregistrée.",
              text_color=MUTED).pack(pady=40)
            return
        for h in reversed(historique):
            carte = ctk.CTkFrame(self.liste_dissonances, fg_color=SURFACE, corner_radius=14,
                                 border_width=2,
                                 border_color=INDIGO_FONCE if h.get("mode") == "action" else SURFACE2)
            carte.pack(fill="x", pady=5, padx=2)
            tete = ctk.CTkFrame(carte, fg_color="transparent")
            tete.pack(fill="x", padx=16, pady=(10, 2))
            mode = h.get("mode", "?")
            couleur = GOOD if mode == "action" else WARN
            L(tete, f"🧠 {mode.upper()}", text_color=couleur,
              font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
            L(tete, h.get("timestamp", "?"), text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(side="right")
            L(carte, h.get("action", "?"), wraplength=900, justify="left", anchor="w",
              font=ctk.CTkFont(size=12)).pack(anchor="w", padx=16, pady=(2, 4))
            if "gain_energie" in h:
                L(carte, f"   Gain d'énergie : {h['gain_energie']:.3f}",
                  text_color=MUTED, font=ctk.CTkFont(size=10)).pack(anchor="w", padx=16, pady=(0, 10))
            else:
                L(carte, "   (simulation, pas d'action réelle)", text_color=MUTED,
                  font=ctk.CTkFont(size=10)).pack(anchor="w", padx=16, pady=(0, 10))

    def action_afficher_dissonances(self):
        if not self.agent or self.agent.demon is None:
            self.toast("⚠️ Démon de dissonance désactivé.", WARN)
            return
        self.afficher_page("dissonances")

    def action_basculer_simulation(self):
        if not self.agent or self.agent.demon is None:
            return
        nouveau = self.agent.basculer_simulation_dissonance()
        if nouveau is not None:
            mode = "SIMULATION" if nouveau else "ACTION RÉELLE"
            couleur = WARN if nouveau else GOOD
            self.toast(f"🧠 Mode : {mode}", couleur)
            self.charger_dissonances()

    def action_pause_dissonance(self):
        if not self.agent or self.agent.demon is None:
            return
        self.agent.pause_dissonance()
        self.toast("⏸️ Démon en pause.", WARN)
        self.charger_dissonances()

    def action_reprendre_dissonance(self):
        if not self.agent or self.agent.demon is None:
            return
        self.agent.reprendre_dissonance()
        self.toast("▶️ Démon relancé.", GOOD)
        self.charger_dissonances()

    # ---------- page : archives ----------
    def _page_archives(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        L(barre, "🗜️  Archives compressées", font=ctk.CTkFont(size=16, weight="bold"),
          text_color=TEAL).pack(side="left", padx=20, pady=14)
        b_note = B(barre, "➕  Nouvelle note", fg_color=TEAL,
                   hover_color=TEAL_FONCE, text_color="#ffffff",
                   command=self.action_archiver_note)
        b_note.pack(side="right", padx=(6, 14), pady=12)
        self.boutons_action.append(b_note)
        b_export = B(barre, "📤  Exporter tout", fg_color=SURFACE2, hover_color=TEAL,
                     text_color=TEXT, command=self.action_exporter_archives)
        b_export.pack(side="right", padx=6)
        self.boutons_action.append(b_export)
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_archives).pack(side="right", padx=6)

        stats_carte = self._carte(page)
        stats_carte.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        self.lbl_archives_stats = L(stats_carte, "…", text_color=MUTED, anchor="w",
                                    font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_archives_stats.pack(fill="x", padx=20, pady=(14, 4))
        self.barre_archives_stats = ctk.CTkProgressBar(stats_carte, height=12, progress_color=TEAL)
        self.barre_archives_stats.set(0)
        self.barre_archives_stats.pack(fill="x", padx=20, pady=(0, 4))
        self.lbl_archives_detail = L(stats_carte, "", text_color=MUTED, anchor="w",
                                     font=ctk.CTkFont(size=11))
        self.lbl_archives_detail.pack(fill="x", padx=20, pady=(0, 14))

        self.liste_archives = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.liste_archives.grid(row=2, column=0, sticky="nsew")
        L(self.liste_archives,
          "Aucune note archivée. Cliquez sur « ➕ Nouvelle note » pour commencer.",
          text_color=MUTED).pack(pady=40)
        return page

    def charger_archives(self):
        if not self.agent:
            return

        def lire():
            return {"stats": self.agent.stats_archives(),
                    "notes": self.agent.lister_archives()}

        self.run_async(lire, self._apres_archives, exclusif=False)

    def _apres_archives(self, data, _sortie):
        if data is None:
            return
        stats = data["stats"]
        notes = data["notes"]
        self.donnees_archives = notes
        self.stats_archives_cache = stats

        # Stats
        self.lbl_archives_stats.configure(
            text=T(f"{stats['total_notes']} note(s) · "
                   f"{stats['total_go']:.2f} Go / {stats['limite_go']:.1f} Go "
                   f"({stats['pourcentage']:.1f} %)"))
        self.barre_archives_stats.set(min(1.0, max(0.0, stats["pourcentage"] / 100.0)))
        couleur = GOOD if stats["pourcentage"] < 85 else (
            WARN if stats["pourcentage"] < 100 else BAD)
        self.barre_archives_stats.configure(progress_color=couleur)
        detail = f"Dossier : {stats['dossier']}"
        if stats.get("par_type"):
            detail += "   ·   " + "  ·  ".join(
                f"{t} : {n}" for t, n in sorted(stats["par_type"].items()))
        self.lbl_archives_detail.configure(text=T(detail))

        for enfant in self.liste_archives.winfo_children():
            enfant.destroy()
        if not notes:
            L(self.liste_archives,
              "Aucune note archivée. Cliquez sur « ➕ Nouvelle note » pour commencer.",
              text_color=MUTED).pack(pady=40)
            return

        for note in notes:
            carte = ctk.CTkFrame(self.liste_archives, fg_color=SURFACE, corner_radius=14,
                                 border_width=1, border_color=TEAL_FONCE)
            carte.pack(fill="x", pady=5, padx=2)
            tete = ctk.CTkFrame(carte, fg_color="transparent")
            tete.pack(fill="x", padx=16, pady=(10, 2))
            L(tete, f"🗜️  {note['titre']}", text_color=TEAL,
              font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            L(tete, note["date_creation"], text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(side="right")
            infos = (f"{note['taille_originale']} → {note['taille_compressee']} o  ·  "
                     f"algo {note['algo']}  ·  type {note['type_note']}")
            L(carte, infos, text_color=MUTED, font=ctk.CTkFont(size=11)).pack(
                anchor="w", padx=16, pady=(0, 2))
            if note.get("preview"):
                apercu = note["preview"].replace("\n", " ")[:200]
                L(carte, apercu, text_color=TEXT, wraplength=900, justify="left",
                  anchor="w", font=ctk.CTkFont(size=12)).pack(
                    anchor="w", padx=16, pady=(0, 4))
            rangee = ctk.CTkFrame(carte, fg_color="transparent")
            rangee.pack(fill="x", padx=16, pady=(0, 10))
            B(rangee, "📖  Lire", width=90, height=26, fg_color=SURFACE2,
              hover_color=ACCENT, text_color=TEXT,
              command=lambda i=note["id"]: self.action_lire_archive(i)).pack(side="left")
            B(rangee, "🗑️  Supprimer", width=110, height=26, fg_color=SURFACE2,
              hover_color=BAD, text_color=TEXT,
              command=lambda i=note["id"]: self.action_supprimer_archive(i)).pack(
                side="left", padx=6)

    def action_archiver_note(self):
        if not self.agent:
            return
        dialogue = ctk.CTkInputDialog(text="Titre de la note :", title="Nouvelle note")
        titre = (dialogue.get_input() or "").strip()
        if not titre:
            return
        dialogue2 = ctk.CTkInputDialog(text="Contenu de la note :", title="Nouvelle note")
        contenu = (dialogue2.get_input() or "").strip()
        if not contenu:
            self.toast("⚠️ Contenu vide, note non archivée.", WARN)
            return

        def travail():
            return self.agent.archiver_note(titre, contenu)

        def apres(res, _sortie):
            note = res["note"]
            gain = (1 - note["taille_compressee"] / max(1, note["taille_originale"])) * 100
            self.toast(
                f"🗜️ Note archivée · {note['taille_originale']} → "
                f"{note['taille_compressee']} o (gain {gain:.1f} %)",
                GOOD)
            purge = res.get("purge") or {}
            if purge.get("purge") and purge.get("supprimees", 0) > 0:
                self.toast(f"♻️ Purge auto : {purge['supprimees']} note(s) supprimée(s).", WARN)
            self.charger_archives()

        self.run_async(travail, apres, "Archivage…")

    def action_lire_archive(self, id_note):
        if not self.agent:
            return

        def travail():
            return self.agent.lire_archive(id_note)

        def apres(contenu, _sortie):
            if contenu is None:
                self.toast("⚠️ Note introuvable ou décompression impossible.", WARN)
                return
            self.fenetre_texte("🗜️ Note archivée", contenu)

        self.run_async(travail, apres, "Lecture…", exclusif=False)

    def action_supprimer_archive(self, id_note):
        if not self.agent:
            return
        if not messagebox.askyesno("Supprimer la note",
                                   "Supprimer définitivement cette note archivée ?"):
            return

        def travail():
            return self.agent.supprimer_archive(id_note)

        def apres(succes, _sortie):
            if succes:
                self.toast("🗑️ Note supprimée.", GOOD)
            else:
                self.toast("⚠️ Note introuvable.", WARN)
            self.charger_archives()

        self.run_async(travail, apres, "Suppression…")

    def action_exporter_archives(self):
        if not self.agent:
            return
        dossier = filedialog.askdirectory(title="Dossier d'export des archives")
        if not dossier:
            return

        def travail():
            return self.agent.archives.exporter_tout(dossier)

        def apres(res, _sortie):
            if not res:
                self.toast("⚠️ Rien à exporter.", WARN)
                return
            self.toast(f"📤 {res['exportees']} note(s) exportée(s) vers {res['dossier']}.", GOOD)

        self.run_async(travail, apres, "Export des archives…")

    # ---------- page : documents ----------
    def _page_documents(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        L(barre, "📚  Documents de Nyx", font=ctk.CTkFont(size=16, weight="bold"),
          text_color=TEAL).pack(side="left", padx=20, pady=14)
        b_lire = B(barre, "📖  Relire un document (spontané)", fg_color=TEAL,
                   hover_color=TEAL_FONCE, text_color="#ffffff",
                   command=self.action_lecture_spontanee)
        b_lire.pack(side="right", padx=(6, 14), pady=12)
        self.boutons_action.append(b_lire)
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_documents).pack(side="right", padx=6)

        self.liste_documents = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.liste_documents.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        L(self.liste_documents,
          "Aucun document. Dis à Nyx « écris-moi un document sur… » pour en créer un.",
          text_color=MUTED).pack(pady=40)
        return page

    def charger_documents(self):
        if not self.agent:
            return

        def lire():
            return list(self.agent.lister_documents())

        self.run_async(lire, self._apres_documents, exclusif=False)

    def _apres_documents(self, docs, _sortie):
        self.donnees_documents = docs or []
        for enfant in self.liste_documents.winfo_children():
            enfant.destroy()
        if not self.donnees_documents:
            L(self.liste_documents,
              "Aucun document. Dis à Nyx « écris-moi un document sur… » pour en créer un.",
              text_color=MUTED).pack(pady=40)
            return

        for d in self.donnees_documents:
            carte = ctk.CTkFrame(self.liste_documents, fg_color=SURFACE, corner_radius=14,
                                 border_width=1, border_color=TEAL_FONCE)
            carte.pack(fill="x", pady=5, padx=2)
            tete = ctk.CTkFrame(carte, fg_color="transparent")
            tete.pack(fill="x", padx=16, pady=(10, 2))
            L(tete, f"📄  {d.nom}", text_color=TEAL,
              font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            L(tete, f"{d.taille} o", text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(side="right")
            L(carte, f"modifié le {d.modifie_le}", text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16, pady=(0, 4))

            rangee = ctk.CTkFrame(carte, fg_color="transparent")
            rangee.pack(fill="x", padx=16, pady=(0, 10))
            B(rangee, "📖  Lire", width=90, height=26, fg_color=SURFACE2,
              hover_color=ACCENT, text_color=TEXT,
              command=lambda n=d.nom: self.action_lire_document(n)).pack(side="left")

    def action_lire_document(self, nom):
        if not self.agent:
            return

        def travail():
            return self.agent.lire_document(nom)

        def apres(contenu, _sortie):
            if contenu is None:
                self.toast(f"⚠️ Document « {nom} » introuvable.", WARN)
                return
            self.fenetre_texte(f"📄 {nom}", contenu)

        self.run_async(travail, apres, "Lecture…", exclusif=False)

    def action_lecture_spontanee(self):
        if not self.agent:
            return
        if self.occupe:
            self.toast("⏳ Une opération est déjà en cours.", WARN)
            return
        if not getattr(self.agent.cfg, "lecture_auto_actif", True):
            self.toast("⚠️ Les lectures spontanées sont désactivées.", WARN)
            return
        if not self.agent.lister_documents():
            self.toast("⚠️ Aucun document à lire. Dis « écris-moi un document sur… ».", WARN)
            return

        def travail():
            return self.agent.lecture_spontanee()

        def apres(resultat, _sortie):
            if not resultat:
                self.toast("⚠️ Aucune lecture n'a pu être faite.", WARN)
                return
            self.toast(f"📖 Nyx a relu « {resultat.get('titre', '?')} ».", TEAL)
            self.maj_emotions()
            self.charger_documents()

        self.run_async(travail, apres, "Nyx relit un document…")

    # ---------- mises à jour ----------
    def maj_emotions(self):
        a = self.agent
        if not a:
            return
        nom, _ = a._calculer_emotion_dominante()
        self.lbl_emoji.configure(text=T(EMOJIS_EMOTIONS.get(nom, "😐")))
        self.lbl_emotion.configure(text=nom)
        valeurs = {"valence": (a.valence + 1) / 2, "activation": a.activation, "dominance": a.dominance}
        textes = {"valence": f"{a.valence:+.2f}", "activation": f"{a.activation:.2f}",
                  "dominance": f"{a.dominance:.2f}"}
        for cle, (barre_p, etiquette) in self.jauges_chat.items():
            barre_p.set(max(0.0, min(1.0, valeurs[cle])))
            etiquette.configure(text=textes[cle])
        self.lbl_stats_chat.configure(text=T(
            f"🧠 {a.hippocampe.count()} souvenirs\n📖 {a.journal_intime.count()} réflexions\n"
            f"📄 {len(a.lister_documents())} documents"))
        self.maj_pulsions()
        self.maj_taille_memoire()
        self._maj_bouton_voix()
        self._maj_bouton_pensees()
        if hasattr(self, "jauges_etat"):
            for cle, (barre_p, etiquette) in self.jauges_etat.items():
                barre_p.set(max(0.0, min(1.0, valeurs[cle])))
                etiquette.configure(text=textes[cle])
            self.lbl_emoji_etat.configure(text=T(EMOJIS_EMOTIONS.get(nom, "😐")))
            self.lbl_emotion_etat.configure(text=nom)

    def maj_pulsions(self):
        a = self.agent
        if not a:
            return
        p = a.pulsions_cache or {}
        fatigue = p.get("fatigue", 0.0)
        envie = p.get("envie_reflexion", 0.0)
        envie_lire = p.get("envie_lire", 0.0)
        coul_fatigue = BAD if fatigue >= a.SEUIL_SOMMEIL_AUTO else (
            WARN if fatigue >= a.SEUIL_ENVIE_DORMIR else ACCENT)
        coul_envie = WARN if envie >= a.SEUIL_REFLEXION else ACCENT
        coul_lire = WARN if envie_lire >= a.SEUIL_ENVIE_LIRE else ACCENT
        for jauges in (self.jauges_pulsions_chat, self.jauges_pulsions_etat):
            for cle, valeur, couleur in (("fatigue", fatigue, coul_fatigue),
                                         ("envie_reflexion", envie, coul_envie),
                                         ("envie_lire", envie_lire, coul_lire)):
                if cle in jauges:
                    barre_p, etiquette = jauges[cle]
                    barre_p.set(max(0.0, min(1.0, valeur)))
                    barre_p.configure(progress_color=couleur)
                    etiquette.configure(text=f"{valeur:.0%}")
        if hasattr(self, "lbl_declin"):
            self.lbl_declin.configure(text=T(
                f"🕳️ {p.get('nb_declin', 0)} souvenir(s) qui s'estompent · "
                f"{p.get('nb_faibles', 0)} presque oublié(s)  ·  "
                f"un sommeil est déclenché à ≥ {a.SEUIL_SOMMEIL_AUTO:.0%} de fatigue "
                f"(si Nyx est inactive), "
                f"une pensée à voix haute à ≥ {a.SEUIL_REFLEXION:.0%} d'envie, "
                f"une relecture à ≥ {a.SEUIL_ENVIE_LIRE:.0%} d'envie de lire"))
        if hasattr(self, "lbl_pipeline") and a.dernier_pipeline:
            self.lbl_pipeline.configure(text=T("\n".join(
                f"{i}. {nom}  ·  {duree:.2f}s" for i, (nom, duree) in enumerate(a.dernier_pipeline, 1))))

    def maj_taille_memoire(self):
        a = self.agent
        if not a:
            return
        try:
            stats = a.verifier_taille_memoire()
        except Exception:
            return
        self.stats_taille = stats
        couleur = couleur_taille(stats)
        emoji = emoji_taille(stats)
        if hasattr(self, "lbl_taille_chat"):
            self.lbl_taille_chat.configure(
                text=T(f"{emoji} {stats.taille_go:.3f} / {stats.limite_go:.2f} Go "
                       f"({stats.fraction * 100:.1f} %)"),
                text_color=couleur,
            )
            self.barre_taille_chat.set(min(1.0, max(0.0, stats.fraction)))
            self.barre_taille_chat.configure(progress_color=couleur)
        if hasattr(self, "lbl_taille_etat"):
            self.lbl_taille_etat.configure(
                text=T(f"{emoji} {stats.taille_go:.3f} Go / {stats.limite_go:.2f} Go  "
                       f"({stats.fraction * 100:.1f} %)"),
                text_color=couleur,
            )
            self.barre_taille_etat.set(min(1.0, max(0.0, stats.fraction)))
            self.barre_taille_etat.configure(progress_color=couleur)
            self.lbl_taille_detail_etat.configure(text=T(
                f"{stats.nb_fichiers} fichier(s) · "
                f"avertissement à {a.memoire_seuil_avertissement * 100:.0f} % · "
                f"purge auto : {'ON' if a.memoire_purge_auto_actif else 'OFF'}"
            ))
        if hasattr(self, "lbl_embedding_chat"):
            if getattr(a, "ef", None) is not None:
                self.lbl_embedding_chat.configure(text=T(f"🔢 {a.embedding_model}"))
            else:
                self.lbl_embedding_chat.configure(text=T("🔢 Chroma par défaut"))
        if hasattr(self, "lbl_voix_chat"):
            if a.voix is None:
                self.lbl_voix_chat.configure(text=T("🔇 Voix non initialisée"))
            else:
                info = a.verifier_voix()
                if info.disponible:
                    self.lbl_voix_chat.configure(text=T("🔊 Piper (Siwis FR)"),
                                                 text_color=GOOD)
                else:
                    self.lbl_voix_chat.configure(text=T("⚠️ Voix indisponible"),
                                                 text_color=BAD)
        try:
            s = a.stats_archives()
            self.stats_archives_cache = s
            couleur_a = couleur_archive(s)
            emoji_a = emoji_archive(s)
            if hasattr(self, "lbl_archives_chat"):
                self.lbl_archives_chat.configure(
                    text=T(f"{emoji_a} {s['total_notes']} note(s) · "
                           f"{s['total_go']:.2f} / {s['limite_go']:.1f} Go "
                           f"({s['pourcentage']:.1f} %)"),
                    text_color=couleur_a,
                )
                self.barre_archives_chat.set(min(1.0, max(0.0, s["pourcentage"] / 100.0)))
                self.barre_archives_chat.configure(progress_color=couleur_a)
            if hasattr(self, "lbl_archives_etat"):
                self.lbl_archives_etat.configure(
                    text=T(f"{emoji_a} {s['total_notes']} note(s) · "
                           f"{s['total_go']:.2f} / {s['limite_go']:.1f} Go "
                           f"({s['pourcentage']:.1f} %)"),
                    text_color=couleur_a,
                )
                self.barre_archives_etat.set(min(1.0, max(0.0, s["pourcentage"] / 100.0)))
                self.barre_archives_etat.configure(progress_color=couleur_a)
        except Exception:
            pass

    def _boucle_ui(self):
        self.maj_pulsions()
        self.maj_taille_memoire()
        self._maj_bouton_voix()
        self._maj_bouton_pensees()
        if self.page_courante == "dissonances" and self.agent and self.agent.demon:
            try:
                self.lbl_dissonance_diag.configure(text=self.agent.demon.diagnostique())
            except Exception:
                pass
        self.after(1500, self._boucle_ui)

    # ---------- événements ----------
    def _evenement_agent(self, genre, texte):
        self.file_ui.put(("call", lambda: self._afficher_evenement(genre, texte)))

    def _afficher_evenement(self, genre, texte):
        if genre == "auto_reflexion":
            self.ajouter_bulle("nyx", texte, genre="pensee")
            self.toast("💭 Nyx pense à voix haute.", ACCENT)
        elif genre == "envie_dormir":
            self.ajouter_bulle("nyx", texte, genre="sommeil")
            self.toast("😴 Nyx a sommeil.", WARN)
        elif genre == "sommeil":
            self._pastille("Nyx dort…", WARN)
            self.toast("💤 Nyx s'endort (NREM + REM)…", WARN)
        elif genre == "reveil":
            self.ajouter_bulle("nyx", texte, genre="reveil")
            self._pastille(self.agent.model_name, GOOD)
            self.toast("🌅 Nyx s'est réveillée.", GOOD)
        elif genre == "avertissement_taille":
            self.ajouter_bulle("nyx", texte, genre="avertissement_taille")
            self.toast("⚠️ Mémoire proche de sa limite.", WARN)
        elif genre == "rappel":
            self.ajouter_bulle("nyx", texte, genre="reveil")
            self.toast("⏰ Rappel d'agenda.", ACCENT)
        elif genre == "interpellation":
            self.ajouter_bulle("nyx", texte, genre="interpellation")
            self.toast("📢 Nyx t'interpelle.", ACCENT)
        elif genre == "reve":
            self.ajouter_bulle("nyx", texte, genre="reve")
            self.toast("🌙 Nyx rêve.", VIOLET)
        elif genre == "dissonance_simulation":
            self.ajouter_bulle("nyx", f"🧠 {texte}", genre="dissonance")
            self.toast("🧠 Dissonance (simulation)", INDIGO)
        elif genre == "dissonance_resolue":
            self.ajouter_bulle("nyx", f"🧠 {texte}", genre="dissonance")
            self.toast("🧠 Dissonance résolue", GOOD)
        elif genre == "archive":
            self.ajouter_bulle("nyx", f"🗜️ {texte}", genre="archive")
            self.toast("🗜️ Archive", TEAL)
        elif genre == "lecture_spontanee":
            self.ajouter_bulle("nyx", texte, genre="lecture_spontanee")
            self.toast("📖 Nyx relit un de ses documents.", ACCENT)
        self.maj_emotions()
        self.rafraichir()

    # ---------- page : mémoire ----------
    def _page_memoire(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        self.filtre_mem = self._option(barre, [f[0] for f in self.FILTRES_MEM], lambda v: self.rendre_memoire())
        self.filtre_mem.pack(side="left", padx=(14, 6), pady=12)
        self.tri_mem = self._option(barre, self.TRIS_MEM, lambda v: self.rendre_memoire(), 160)
        self.tri_mem.pack(side="left", padx=6)
        self.entree_mem = ctk.CTkEntry(barre, placeholder_text=T("🔍 Rechercher…"), width=220,
                                       fg_color=SURFACE2, border_width=0, placeholder_text_color=MUTED)
        self.entree_mem.pack(side="left", padx=6)
        self.entree_mem.bind("<KeyRelease>", lambda e: self.rendre_memoire())
        B(barre, "🗑️  Supprimer la sélection", fg_color=SURFACE2, hover_color=BAD, text_color=TEXT,
          command=self.supprimer_souvenirs).pack(side="right", padx=(6, 14))
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_memoire).pack(side="right", padx=6)
        self.lbl_stats_mem = L(page, "", text_color=MUTED, anchor="w")
        self.lbl_stats_mem.grid(row=1, column=0, sticky="ew", padx=6, pady=(8, 4))
        cadre = self._carte(page)
        cadre.grid(row=2, column=0, sticky="nsew")
        cadre.grid_columnconfigure(0, weight=1)
        cadre.grid_rowconfigure(0, weight=1)
        colonnes = (("date", "Date", 170), ("type", "Type", 130), ("force", "Force synaptique", 170),
                    ("react", "×", 50), ("texte", "Souvenir", 600))
        self.arbre = ttk.Treeview(cadre, columns=[c[0] for c in colonnes], show="headings",
                                  style="Nyx.Treeview", selectmode="extended")
        for cle, titre, largeur in colonnes:
            self.arbre.heading(cle, text=titre, anchor="w")
            self.arbre.column(cle, width=largeur, anchor="w", stretch=(cle == "texte"))
        self.arbre.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        defil = ctk.CTkScrollbar(cadre, command=self.arbre.yview)
        defil.grid(row=0, column=1, sticky="ns", pady=10, padx=(0, 6))
        self.arbre.configure(yscrollcommand=defil.set)
        self.arbre.bind("<<TreeviewSelect>>", self._detail_souvenir)
        self.detail_mem = ctk.CTkTextbox(page, height=96, fg_color=SURFACE, corner_radius=14, wrap="word")
        self.detail_mem.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.detail_mem.configure(state="disabled")
        return page

    def _style_arbre(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        s = sombre()
        fond, texte, entete = ("#161923", "#e7e9f3", "#1f2331") if s else ("#ffffff", "#191c27", "#e9ebf2")
        style.layout("Nyx.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.configure("Nyx.Treeview", background=fond, fieldbackground=fond, foreground=texte,
                        borderwidth=0, rowheight=30)
        style.configure("Nyx.Treeview.Heading", background=entete, foreground=texte, relief="flat",
                        borderwidth=0, padding=(8, 7))
        style.map("Nyx.Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        style.map("Nyx.Treeview.Heading", background=[("active", entete)])
        self.arbre.tag_configure("marquant", foreground="#f08a4b")
        self.arbre.tag_configure("concept", foreground="#5aa9ff")
        self.arbre.tag_configure("onirique", foreground=VIOLET)
        self.arbre.tag_configure("faible", foreground="#8a90a6")

    def charger_memoire(self):
        self.run_async(self._lire_memoire, self._apres_memoire, exclusif=False)

    def _lire_memoire(self):
        a = self.agent
        if a.hippocampe.count() == 0:
            return []
        data = a.hippocampe.get()
        lignes = []
        for doc_id, doc, meta in zip(data['ids'], data['documents'], data['metadatas']):
            try:
                n = a._normaliser_meta(meta)
                force = a._force_effective(meta)
            except (ValueError, TypeError):
                n, force = a._normaliser_meta(None), 0.0
            if n['type'] == 'concept_consolidé':
                cat = "concept"
            elif n['type'] == 'souvenir_onirique':
                cat = "onirique"
            elif a._est_marquant(n):
                cat = "marquant"
            else:
                cat = "brut"
            faible = force < a.SEUIL_OUBLI and cat not in ("marquant", "onirique")
            lignes.append({"id": doc_id, "texte": doc or "", "date": n['created_at'],
                           "acces": n['dernier_acces'], "cat": cat, "force": force,
                           "react": n['reactivations'], "faible": faible,
                           "valence": n['valence'], "activation": n['activation']})
        return lignes

    def _apres_memoire(self, lignes, _sortie):
        self.donnees_memoire = lignes
        self.rendre_memoire()
        if self.page_courante == "etat":
            self.rendre_etat()

    def rendre_memoire(self):
        cles = {T(lbl): cle for lbl, cle in self.FILTRES_MEM}
        filtre = cles.get(self.filtre_mem.get(), "tous")
        recherche = self.entree_mem.get().strip().lower()
        lignes = []
        for l in self.donnees_memoire:
            if recherche and recherche not in l["texte"].lower():
                continue
            if filtre == "faible" and not l["faible"]:
                continue
            if filtre in ("marquant", "concept", "brut", "onirique") and l["cat"] != filtre:
                continue
            lignes.append(l)
        tri = self.tri_mem.get()
        cle_tri = {"Plus récents": ("date", True), "Plus anciens": ("date", False),
                   "Plus forts": ("force", True), "Plus faibles": ("force", False),
                   "Plus réactivés": ("react", True)}[tri]
        lignes.sort(key=lambda l: l[cle_tri[0]], reverse=cle_tri[1])
        ancienne_selection = set(self.arbre.selection())
        self.arbre.delete(*self.arbre.get_children())
        etiquettes = {"marquant": "🔥 Marquant", "concept": "🧱 Concept",
                      "brut": "💭 Souvenir", "onirique": "🌙 Onirique"}
        for l in lignes:
            tag = "faible" if l["faible"] else (l["cat"] if l["cat"] != "brut" else "")
            self.arbre.insert("", "end", iid=l["id"], tags=(tag,), values=(
                l["date"], T(etiquettes[l["cat"]]), f"{barre_texte(l['force'])}  {l['force']:.2f}",
                l["react"], l["texte"].replace("\n", " ")[:160]))
        restante = [i for i in self.arbre.get_children() if i in ancienne_selection]
        if restante:
            self.arbre.selection_set(restante)
        self._detail_souvenir()
        d = self.donnees_memoire
        self.lbl_stats_mem.configure(text=T(
            f"🧠 {len(d)} souvenirs   ·   🔥 {sum(1 for l in d if l['cat'] == 'marquant')} marquants   ·   "
            f"🧱 {sum(1 for l in d if l['cat'] == 'concept')} concepts   ·   "
            f"🌙 {sum(1 for l in d if l['cat'] == 'onirique')} oniriques   ·   "
            f"💨 {sum(1 for l in d if l['faible'])} faibles   ·   affichés : {len(lignes)}"))

    def _detail_souvenir(self, _event=None):
        selection = self.arbre.selection()
        self.detail_mem.configure(state="normal")
        self.detail_mem.delete("1.0", "end")
        if selection:
            l = next((x for x in self.donnees_memoire if x["id"] == selection[0]), None)
            if l:
                self.detail_mem.insert("end", l["texte"] + "\n\n")
                self.detail_mem.insert("end", f"Créé : {l['date']}  ·  dernier accès : {l['acces']}  ·  "
                                              f"force {l['force']:.2f}  ·  ×{l['react']}  ·  "
                                              f"valence {l['valence']:+.2f}  activation {l['activation']:.2f}")
        self.detail_mem.configure(state="disabled")

    def supprimer_souvenirs(self):
        ids = list(self.arbre.selection())
        if not ids:
            self.toast("Sélectionnez d'abord un ou plusieurs souvenirs.", WARN)
            return
        if not messagebox.askyesno("Supprimer", f"Supprimer définitivement {len(ids)} souvenir(s) ?"):
            return
        self.run_async(lambda: self.agent.hippocampe.delete(ids=ids),
                       lambda r, s: (self.toast(f"🗑️ {len(ids)} souvenir(s) supprimé(s).", GOOD),
                                     self.charger_memoire()), "Suppression…")

    # ---------- page : journal ----------
    def _page_journal(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        self.filtre_journal = self._option(barre, ["Tous les types"], lambda v: self.rendre_journal(), 190)
        self.filtre_journal.pack(side="left", padx=(14, 6), pady=12)
        self.ordre_journal = self._option(barre, ["Plus récentes d'abord", "Plus anciennes d'abord"],
                                          lambda v: self.rendre_journal(), 190)
        self.ordre_journal.pack(side="left", padx=6)
        self.entree_journal = ctk.CTkEntry(barre, placeholder_text=T("🔍 Rechercher…"), width=200,
                                           fg_color=SURFACE2, border_width=0, placeholder_text_color=MUTED)
        self.entree_journal.pack(side="left", padx=6)
        self.entree_journal.bind("<KeyRelease>", lambda e: self.rendre_journal())
        b_id = B(barre, "🪞  Me décrire", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
                 command=self.action_identite)
        b_id.pack(side="right", padx=(6, 14))
        b_new = B(barre, "✍️  Nouvelle réflexion", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
                  command=self.nouvelle_reflexion)
        b_new.pack(side="right", padx=6)
        B(barre, "🔄  Actualiser", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.charger_journal).pack(side="right", padx=6)
        self.boutons_action += [b_id, b_new]
        self.liste_journal = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.liste_journal.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        return page

    def charger_journal(self):
        def lire():
            return [{"epoch": e, "texte": d, "meta": m or {}}
                    for e, d, m in self.agent._entrees_journal_triees()]
        self.run_async(lire, self._apres_journal, exclusif=False)

    def _apres_journal(self, entrees, _sortie):
        self.donnees_journal = entrees
        types = sorted({e["meta"].get("type_evenement", "?") for e in entrees})
        self.filtre_journal.configure(values=[T("Tous les types")] + types)
        if self.filtre_journal.get() not in [T("Tous les types")] + types:
            self.filtre_journal.set(T("Tous les types"))
        self.rendre_journal()

    def rendre_journal(self):
        for enfant in self.liste_journal.winfo_children():
            enfant.destroy()
        filtre = self.filtre_journal.get()
        recherche = self.entree_journal.get().strip().lower()
        entrees = [e for e in self.donnees_journal
                   if (filtre == T("Tous les types") or e["meta"].get("type_evenement", "?") == filtre)
                   and (not recherche or recherche in e["texte"].lower())]
        entrees.sort(key=lambda e: e["epoch"],
                     reverse=self.ordre_journal.get().startswith("Plus récentes"))
        if not entrees:
            L(self.liste_journal, "Aucune entrée.", text_color=MUTED).pack(pady=40)
            return
        for e in entrees[:200]:
            meta = e["meta"]
            type_ev = meta.get("type_evenement", "?")
            emotion = meta.get("emotion", "?")
            est_reve = type_ev == "reve"
            est_dissonance = type_ev == "dissonance"
            est_archive = type_ev == "archive_ecrite"
            est_lecture = type_ev in ("document_lu", "lecture_spontanee")

            if est_reve:
                carte = ctk.CTkFrame(self.liste_journal, fg_color=SURFACE, corner_radius=14,
                                     border_width=2, border_color=VIOLET_FONCE)
            elif est_dissonance:
                carte = ctk.CTkFrame(self.liste_journal, fg_color=SURFACE, corner_radius=14,
                                     border_width=2, border_color=INDIGO_FONCE)
            elif est_archive:
                carte = ctk.CTkFrame(self.liste_journal, fg_color=SURFACE, corner_radius=14,
                                     border_width=2, border_color=TEAL_FONCE)
            elif est_lecture:
                carte = ctk.CTkFrame(self.liste_journal, fg_color=SURFACE, corner_radius=14,
                                     border_width=2, border_color=TEAL_FONCE)
            else:
                carte = self._carte(self.liste_journal)
            carte.pack(fill="x", pady=5, padx=2)

            tete = ctk.CTkFrame(carte, fg_color="transparent")
            tete.pack(fill="x", padx=16, pady=(12, 2))
            prefix = ("🌙 " if est_reve else
                      ("🧠 " if est_dissonance else
                       ("🗜️ " if est_archive else
                        ("📖 " if est_lecture else ""))))
            couleur_emotion = (VIOLET if est_reve else
                               (INDIGO if est_dissonance else
                                (TEAL if (est_archive or est_lecture) else ACCENT)))
            L(tete, f"{prefix}{EMOJIS_EMOTIONS.get(emotion, '📖')}  {emotion}",
              text_color=couleur_emotion, font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            L(tete, f"  {type_ev}  ", fg_color=SURFACE2, corner_radius=8,
              text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left", padx=10)
            L(tete, meta.get("timestamp", "?"), text_color=MUTED,
              font=ctk.CTkFont(size=11)).pack(side="right")

            hauteur = self._estimer_hauteur_texte(e["texte"], largeur=880, font_size=13)
            boite = ctk.CTkTextbox(carte, height=hauteur, fg_color="transparent",
                                   border_width=0, wrap="word", text_color=TEXT,
                                   font=ctk.CTkFont(size=13), activate_scrollbars=True)
            boite.pack(fill="x", padx=16, pady=(2, 14))
            boite.insert("1.0", e["texte"])
            boite.configure(state="disabled")
            try:
                boite._textbox.configure(background=couleur_fond(SURFACE), borderwidth=0,
                                         highlightthickness=0, selectbackground=ACCENT,
                                         selectforeground="#ffffff")
            except Exception:
                pass

    def nouvelle_reflexion(self):
        if not self.agent:
            return
        dialogue = ctk.CTkInputDialog(text="Sur quel événement ou sujet Nyx doit-elle écrire ?",
                                      title="Nouvelle réflexion")
        contexte = (dialogue.get_input() or "").strip()
        if not contexte:
            return
        self.run_async(lambda: self.agent._ecrire_dans_journal(contexte, "manuel"),
                       lambda r, s: (self.toast("✍️ Réflexion ajoutée." if r else
                                                "⚠️ Réflexion ignorée.", GOOD if r else WARN),
                                     self.charger_journal()),
                       "Nyx écrit dans son journal…")

    # ---------- page : état & émotions ----------
    def _page_etat(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure((0, 1), weight=1)
        haut = self._carte(page)
        haut.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.lbl_emoji_etat = L(haut, "😐", font=ctk.CTkFont(size=46))
        self.lbl_emoji_etat.pack(side="left", padx=(24, 12), pady=16)
        bloc = ctk.CTkFrame(haut, fg_color="transparent")
        bloc.pack(side="left", pady=16)
        self.lbl_emotion_etat = L(bloc, "Neutralité", text_color=ACCENT,
                                  font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_emotion_etat.pack(anchor="w")
        L(bloc, "Émotion dominante du système limbique (modèle VAD)", text_color=MUTED).pack(anchor="w")
        self.jauges_etat = {}
        grille = ctk.CTkFrame(haut, fg_color="transparent")
        grille.pack(side="right", padx=24, pady=16)
        for i, (cle, nom) in enumerate((("valence", "Valence"), ("activation", "Activation"),
                                        ("dominance", "Dominance"))):
            L(grille, nom, text_color=MUTED, anchor="w", width=90).grid(
                row=i, column=0, pady=4, sticky="w")
            barre_p = ctk.CTkProgressBar(grille, width=240, height=9, progress_color=ACCENT)
            barre_p.set(0.5)
            barre_p.grid(row=i, column=1, padx=10)
            valeur = L(grille, "0.00", width=50)
            valeur.grid(row=i, column=2)
            self.jauges_etat[cle] = (barre_p, valeur)

        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.grid(row=1, column=0, columnspan=2, sticky="ew", pady=12)
        self.cartes_stats = {}
        cles = (("souvenirs", "🧠 Souvenirs"), ("force", "💪 Force moyenne"),
                ("marquants", "🔥 Marquants"), ("concepts", "🧱 Concepts"),
                ("oniriques", "🌙 Oniriques"), ("faibles", "💨 Faibles"),
                ("journal", "📖 Journal"), ("feedbacks", "🎯 Feedbacks"),
                ("dissonances", "🧠 Dissonances"), ("archives", "🗜️ Archives"),
                ("documents", "📚 Documents"))
        for i, (cle, titre) in enumerate(cles):
            stats.grid_columnconfigure(i, weight=1)
            carte = self._carte(stats)
            carte.grid(row=0, column=i, padx=2, sticky="ew")
            L(carte, titre, text_color=MUTED, font=ctk.CTkFont(size=9)).pack(pady=(12, 0))
            valeur = L(carte, "–", font=ctk.CTkFont(size=20, weight="bold"))
            valeur.pack(pady=(0, 12))
            self.cartes_stats[cle] = valeur

        # Carte voix
        voix_carte = self._carte(page)
        voix_carte.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_v = ctk.CTkFrame(voix_carte, fg_color="transparent")
        tete_v.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_v, "🗣️  Voix de Nyx (Piper)", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        B(tete_v, "▶️  Tester", width=100, height=28, fg_color=SURFACE2, hover_color=ACCENT,
          text_color=TEXT, command=self.action_tester_voix).pack(side="right", padx=4)
        B(tete_v, "🔄  Recharger", width=110, height=28, fg_color=SURFACE2, hover_color=ACCENT,
          text_color=TEXT, command=self.action_recharger_voix).pack(side="right", padx=4)
        B(tete_v, "📊  État", width=100, height=28, fg_color=SURFACE2, hover_color=ACCENT,
          text_color=TEXT, command=self.action_etat_voix).pack(side="right", padx=4)
        self.lbl_voix_etat = L(voix_carte, "…", text_color=MUTED, anchor="w",
                               font=ctk.CTkFont(size=12))
        self.lbl_voix_etat.pack(fill="x", padx=20, pady=(4, 14))

        # Carte rêves
        reves_carte = self._carte(page)
        reves_carte.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_r = ctk.CTkFrame(reves_carte, fg_color="transparent")
        tete_r.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_r, "🌙  Rêves", font=ctk.CTkFont(size=14, weight="bold"),
          text_color=VIOLET).pack(side="left")
        B(tete_r, "🌙  Rêver (Ctrl+R)", width=160, height=28, fg_color=VIOLET,
          hover_color=VIOLET_FONCE, text_color="#ffffff",
          command=self.action_rever).pack(side="right", padx=4)
        B(tete_r, "📜  Voir", width=100, height=28, fg_color=SURFACE2, hover_color=VIOLET,
          text_color=TEXT, command=lambda: self.afficher_page("reves")).pack(side="right", padx=4)
        self.lbl_reves_etat = L(reves_carte, "…", text_color=MUTED, anchor="w",
                                font=ctk.CTkFont(size=12))
        self.lbl_reves_etat.pack(fill="x", padx=20, pady=(4, 14))

        # Carte dissonance
        diss_carte = self._carte(page)
        diss_carte.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_d = ctk.CTkFrame(diss_carte, fg_color="transparent")
        tete_d.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_d, "🧠  Dissonances", font=ctk.CTkFont(size=14, weight="bold"),
          text_color=INDIGO).pack(side="left")
        B(tete_d, "🧠  Voir (Ctrl+D)", width=160, height=28, fg_color=INDIGO,
          hover_color=INDIGO_FONCE, text_color="#ffffff",
          command=self.action_afficher_dissonances).pack(side="right", padx=4)
        self.lbl_diss_etat = L(diss_carte, "…", text_color=MUTED, anchor="w",
                               font=ctk.CTkFont(size=12))
        self.lbl_diss_etat.pack(fill="x", padx=20, pady=(4, 14))

        # Carte archives
        archives_carte = self._carte(page)
        archives_carte.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_a = ctk.CTkFrame(archives_carte, fg_color="transparent")
        tete_a.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_a, "🗜️  Archives compressées", font=ctk.CTkFont(size=14, weight="bold"),
          text_color=TEAL).pack(side="left")
        B(tete_a, "🗜️  Voir (Ctrl+A)", width=160, height=28, fg_color=TEAL,
          hover_color=TEAL_FONCE, text_color="#ffffff",
          command=lambda: self.afficher_page("archives")).pack(side="right", padx=4)
        B(tete_a, "📤  Exporter tout", width=140, height=28, fg_color=SURFACE2,
          hover_color=TEAL, text_color=TEXT,
          command=self.action_exporter_archives).pack(side="right", padx=4)
        self.lbl_archives_etat = L(archives_carte, "…", text_color=MUTED, anchor="w",
                                   font=ctk.CTkFont(size=12))
        self.lbl_archives_etat.pack(fill="x", padx=20, pady=(4, 4))
        self.barre_archives_etat = ctk.CTkProgressBar(archives_carte, height=10, progress_color=TEAL)
        self.barre_archives_etat.set(0)
        self.barre_archives_etat.pack(fill="x", padx=20, pady=(0, 14))

        # Carte documents
        docs_carte = self._carte(page)
        docs_carte.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_doc = ctk.CTkFrame(docs_carte, fg_color="transparent")
        tete_doc.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_doc, "📚  Documents", font=ctk.CTkFont(size=14, weight="bold"),
          text_color=TEAL).pack(side="left")
        B(tete_doc, "📚  Voir (Ctrl+L)", width=160, height=28, fg_color=TEAL,
          hover_color=TEAL_FONCE, text_color="#ffffff",
          command=lambda: self.afficher_page("documents")).pack(side="right", padx=4)
        B(tete_doc, "📖  Relire un doc", width=140, height=28, fg_color=SURFACE2,
          hover_color=TEAL, text_color=TEXT,
          command=self.action_lecture_spontanee).pack(side="right", padx=4)
        self.lbl_docs_etat = L(docs_carte, "…", text_color=MUTED, anchor="w",
                               font=ctk.CTkFont(size=12))
        self.lbl_docs_etat.pack(fill="x", padx=20, pady=(4, 14))

        # Carte taille mémoire
        taille = self._carte(page)
        taille.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tete_t = ctk.CTkFrame(taille, fg_color="transparent")
        tete_t.pack(fill="x", padx=20, pady=(14, 4))
        L(tete_t, "💾  Mémoire disque", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        B(tete_t, "📊  Vérifier", width=110, height=28, fg_color=SURFACE2, hover_color=ACCENT,
          text_color=TEXT, command=self.action_verifier_taille).pack(side="right", padx=4)
        b_purge = B(tete_t, "💾  Purger maintenant", width=160, height=28, fg_color=WARN,
                    hover_color="#d18a26", text_color="#ffffff", command=self.action_purger_memoire)
        b_purge.pack(side="right", padx=4)
        self.boutons_action.append(b_purge)
        self.lbl_taille_etat = L(taille, "…", text_color=MUTED, anchor="w",
                                 font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_taille_etat.pack(fill="x", padx=20, pady=(4, 4))
        self.barre_taille_etat = ctk.CTkProgressBar(taille, height=12, progress_color=GOOD)
        self.barre_taille_etat.set(0)
        self.barre_taille_etat.pack(fill="x", padx=20, pady=(0, 4))
        self.lbl_taille_detail_etat = L(taille, "", text_color=MUTED, anchor="w",
                                        font=ctk.CTkFont(size=11))
        self.lbl_taille_detail_etat.pack(fill="x", padx=20, pady=(0, 14))

        puls = self._carte(page)
        puls.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.jauges_pulsions_etat = {}
        rangee = ctk.CTkFrame(puls, fg_color="transparent")
        rangee.pack(fill="x", padx=20, pady=(14, 4))
        for cle, nom in (("fatigue", "😴  Fatigue"),
                         ("envie_reflexion", "💭  Envie de penser"),
                         ("envie_lire", "📖  Envie de lire")):
            bloc = ctk.CTkFrame(rangee, fg_color="transparent")
            bloc.pack(side="left", expand=True, fill="x", padx=(0, 12))
            tete = ctk.CTkFrame(bloc, fg_color="transparent")
            tete.pack(fill="x")
            L(tete, nom, font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            valeur = L(tete, "0%")
            valeur.pack(side="right")
            barre_p = ctk.CTkProgressBar(bloc, height=10, progress_color=ACCENT)
            barre_p.set(0)
            barre_p.pack(fill="x", pady=(6, 0))
            self.jauges_pulsions_etat[cle] = (barre_p, valeur)
        self.lbl_declin = L(puls, "", text_color=MUTED, anchor="w", justify="left", wraplength=1000)
        self.lbl_declin.pack(fill="x", padx=20, pady=(2, 14))

        histo = self._carte(page)
        histo.grid(row=9, column=0, sticky="nsew", padx=(0, 6))
        page.grid_rowconfigure(9, weight=1)
        L(histo, "Répartition des forces synaptiques",
          font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=18, pady=(14, 0))
        self.canvas_histo = tk.Canvas(histo, height=220, highlightthickness=0, bd=0)
        self.canvas_histo.pack(fill="both", expand=True, padx=18, pady=14)
        self.canvas_histo.bind("<Configure>", lambda e: self.dessiner_histo())

        actions = self._carte(page)
        actions.grid(row=9, column=1, sticky="nsew", padx=(6, 0))
        L(actions, "Actions rapides", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=18, pady=(14, 8))
        for texte, cmd in (("💤  Dormir", self.action_dormir), ("🌙  Rêver", self.action_rever),
                           ("🧠  Dissonances", self.action_afficher_dissonances),
                           ("🗜️  Archives", lambda: self.afficher_page("archives")),
                           ("📚  Documents", lambda: self.afficher_page("documents")),
                           ("📖  Relire un doc", self.action_lecture_spontanee),
                           ("🔄  Actualiser", self.charger_etat),
                           ("🪞  Identité", self.action_identite),
                           ("💾  Purger mémoire", self.action_purger_memoire),
                           ("⚡  Check rapide", lambda: self.action_check(True))):
            b = B(actions, texte, height=36, fg_color=SURFACE2, hover_color=ACCENT,
                  text_color=TEXT, command=cmd)
            b.pack(fill="x", padx=18, pady=4)
            if cmd != self.charger_etat:
                self.boutons_action.append(b)
        self.lbl_seuils = L(actions, "", text_color=MUTED, justify="left")
        self.lbl_seuils.pack(anchor="w", padx=18, pady=14)
        self._bins = [0] * 5
        return page

    def charger_etat(self):
        self.maj_emotions()
        self.charger_memoire()

    def rendre_etat(self):
        a, d = self.agent, self.donnees_memoire
        forces = [l["force"] for l in d]
        self.cartes_stats["souvenirs"].configure(text=str(len(d)))
        self.cartes_stats["force"].configure(
            text=f"{sum(forces) / len(forces):.2f}" if forces else "–")
        self.cartes_stats["marquants"].configure(
            text=str(sum(1 for l in d if l["cat"] == "marquant")))
        self.cartes_stats["concepts"].configure(
            text=str(sum(1 for l in d if l["cat"] == "concept")))
        self.cartes_stats["oniriques"].configure(
            text=str(sum(1 for l in d if l["cat"] == "onirique")))
        self.cartes_stats["faibles"].configure(text=str(sum(1 for l in d if l["faible"])))
        self.cartes_stats["journal"].configure(text=str(a.journal_intime.count()))
        self.cartes_stats["feedbacks"].configure(text=str(a.memoire_renforcement.count()))
        if a.demon:
            self.cartes_stats["dissonances"].configure(
                text=str(len(a.demon.etat.historique_resolutions)))
        try:
            self.cartes_stats["archives"].configure(
                text=str(a.stats_archives()["total_notes"]))
        except Exception:
            self.cartes_stats["archives"].configure(text="–")
        try:
            self.cartes_stats["documents"].configure(
                text=str(len(a.lister_documents())))
        except Exception:
            self.cartes_stats["documents"].configure(text="–")
        self._bins = [0] * 5
        for f in forces:
            self._bins[min(4, int(f * 5))] += 1
        self.lbl_seuils.configure(text=(
            f"Seuil d'oubli : {a.SEUIL_OUBLI}\nDecay : {a.TAUX_DECAY}/h\n"
            f"Distance de rappel : {a.SEUIL_DISTANCE}\nModèle chat : {a.model_name}\n"
            f"Embedding : {a.embedding_model if getattr(a, 'ef', None) else 'Chroma défaut'}"))

        # Carte voix
        if hasattr(self, "lbl_voix_etat") and self.agent.voix is not None:
            cfg = a.cfg
            info = a.verifier_voix()
            etat = "🔊 ON" if cfg.voix_actif else "🔇 OFF"
            pen = "💭 parlées" if cfg.voix_pensees else "💭 muettes"
            if info.disponible:
                self.lbl_voix_etat.configure(
                    text=T(f"{etat} · {pen} · Piper Siwis\n{info.message}"),
                    text_color=GOOD)
            else:
                self.lbl_voix_etat.configure(text=T(f"{etat} · {info.message}"),
                                             text_color=BAD)
        # Carte rêves
        if hasattr(self, "lbl_reves_etat"):
            cfg = a.cfg
            if cfg.reve_actif:
                nb_reves = sum(1 for e in a._entrees_journal_triees()
                               if e[2].get("type_evenement") == "reve")
                self.lbl_reves_etat.configure(
                    text=T(f"✅ 2 rêves par sommeil (T dynamique 3 phases)\n"
                           f"   {nb_reves} rêve(s) enregistré(s) dans le journal"),
                    text_color=VIOLET)
            else:
                self.lbl_reves_etat.configure(text=T("❌ Rêves désactivés dans les réglages"),
                                              text_color=MUTED)
        # Carte dissonance
        if hasattr(self, "lbl_diss_etat") and self.agent.demon is not None:
            e = self.agent.demon.etat
            mode = "SIMULATION" if e.mode_simulation else "ACTION RÉELLE"
            couleur = WARN if e.mode_simulation else GOOD
            etat = "⏸️ EN PAUSE" if e.en_pause else ("✅ ACTIF" if e.actif else "❌ DÉSACTIVÉ")
            self.lbl_diss_etat.configure(
                text=T(f"{etat} · Mode : {mode}\n"
                       f"   D = {e.dissonance_actuelle:.2f} / seuil {e.seuil}  ·  "
                       f"Actions : {e.actions_heure}/h  ·  "
                       f"Résolutions : {len(e.historique_resolutions)}"),
                text_color=couleur)
        elif hasattr(self, "lbl_diss_etat"):
            self.lbl_diss_etat.configure(text=T("❌ Démon désactivé"), text_color=MUTED)

        # Carte archives
        if hasattr(self, "lbl_archives_etat"):
            try:
                s = a.stats_archives()
                self.stats_archives_cache = s
                couleur = couleur_archive(s)
                emoji = emoji_archive(s)
                self.lbl_archives_etat.configure(
                    text=T(f"{emoji} {s['total_notes']} note(s) · "
                           f"{s['total_go']:.2f} / {s['limite_go']:.1f} Go "
                           f"({s['pourcentage']:.1f} %)"),
                    text_color=couleur)
                self.barre_archives_etat.set(min(1.0, max(0.0, s["pourcentage"] / 100.0)))
                self.barre_archives_etat.configure(progress_color=couleur)
            except Exception:
                pass

        # Carte documents
        if hasattr(self, "lbl_docs_etat"):
            try:
                docs = a.lister_documents()
                nb = len(docs)
                if nb:
                    dernier = docs[0]
                    self.lbl_docs_etat.configure(
                        text=T(f"{nb} document(s) · dernier : {dernier.nom} "
                               f"({dernier.taille} o)"),
                        text_color=TEAL)
                else:
                    self.lbl_docs_etat.configure(
                        text=T("Aucun document. Dis « écris-moi un document sur… »."),
                        text_color=MUTED)
            except Exception:
                pass

        self.dessiner_histo()
        self.maj_taille_memoire()

    def dessiner_histo(self):
        c = self.canvas_histo
        c.delete("all")
        c.configure(bg="#161923" if sombre() else "#ffffff")
        largeur, hauteur = max(c.winfo_width(), 200), max(c.winfo_height(), 120)
        maxi = max(self._bins) or 1
        texte = "#e7e9f3" if sombre() else "#191c27"
        pas = largeur / 5
        for i, n in enumerate(self._bins):
            h = (hauteur - 50) * n / maxi
            x0, x1 = i * pas + 18, (i + 1) * pas - 18
            c.create_rectangle(x0, hauteur - 28 - h, x1, hauteur - 28, fill=ACCENT, outline="")
            c.create_text((x0 + x1) / 2, hauteur - 40 - h, text=str(n), fill=texte)
            c.create_text((x0 + x1) / 2, hauteur - 10, text=f"{i / 5:.1f} – {(i + 1) / 5:.1f}",
                          fill="#8a90a6")

    # ---------- page : logs ----------
    def _page_logs(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        barre = self._carte(page)
        barre.grid(row=0, column=0, sticky="ew")
        self.selection_logs = self._option(barre, ["📟 Console (session en cours)"], self._choisir_log, 330)
        self.selection_logs.pack(side="left", padx=(14, 6), pady=12)
        for texte, cmd, couleur in (("🔎  Check complet", lambda: self.action_check(False), ACCENT),
                                    ("⚡  Check rapide", lambda: self.action_check(True), SURFACE2)):
            b = B(barre, texte, fg_color=couleur, hover_color=ACCENT_HOVER,
                  text_color="#ffffff" if couleur == ACCENT else TEXT, command=cmd)
            b.pack(side="right", padx=6 if texte.startswith("⚡") else (6, 14))
            self.boutons_action.append(b)
        B(barre, "📋  Copier", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=lambda: self._copier(self.txt_logs.get("1.0", "end"))).pack(side="right", padx=6)
        B(barre, "📂  Dossier", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=lambda: self.ouvrir_dossier("logs")).pack(side="right", padx=6)
        B(barre, "🔄", width=40, fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=self.charger_liste_logs).pack(side="right", padx=6)
        self.bandeau = L(page, "", anchor="w", corner_radius=10, fg_color="transparent")
        self.bandeau.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.txt_logs = ctk.CTkTextbox(page, fg_color=SURFACE, corner_radius=14, wrap="none",
                                       font=ctk.CTkFont(family="Courier", size=12))
        self.txt_logs.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        for tag, couleur in (("ok", GOOD), ("avert", WARN), ("mal", BAD),
                             ("titre", ACCENT), ("gris", "#8a90a6")):
            self.txt_logs.tag_config(tag, foreground=couleur)
        self.txt_logs.configure(state="disabled")
        return page

    def _lister_logs(self):
        if not os.path.isdir("logs"):
            return []
        noms = [n for n in os.listdir("logs") if n.endswith(".txt")]
        return sorted(noms, key=lambda n: os.path.getmtime(os.path.join("logs", n)), reverse=True)

    def charger_liste_logs(self):
        console = T("📟 Console (session en cours)")
        actuel = self.selection_logs.get()
        noms = self._lister_logs()
        self.selection_logs.configure(values=[console] + noms)
        if actuel not in [console] + noms:
            actuel = console
        self.selection_logs.set(actuel)
        self._choisir_log(actuel)

    def _choisir_log(self, valeur):
        if valeur.startswith(T("📟")):
            self.console_visible = self.page_courante == "logs"
            self._ecrire_logs("\n".join(self.console.lignes))
            self.bandeau.configure(
                text=T("  📟 Console en direct : tout ce que Nyx affiche apparaît ici."),
                text_color=MUTED)
            return
        self.console_visible = False
        try:
            with open(os.path.join("logs", valeur), encoding="utf-8") as f:
                contenu = f.read()
        except OSError as e:
            self.toast(f"Lecture impossible : {e}", BAD)
            return
        self._ecrire_logs(contenu, en_bas=False)
        verdict = next((l for l in contenu.splitlines() if l.startswith("VERDICT")), "")
        bilan = next((l for l in contenu.splitlines() if l.startswith("BILAN")), "")
        if verdict:
            couleur = GOOD if "✅" in verdict else (WARN if "⚠" in verdict else BAD)
            self.bandeau.configure(text=T(f"  {verdict}    ·    {bilan}"), text_color=couleur)
        else:
            self.bandeau.configure(text=f"  {valeur}", text_color=MUTED)

    def _tag_ligne(self, ligne):
        s = ligne.lstrip()
        if s.startswith(("[OK", "✅")) or "✅" in s:
            return "ok"
        if s.startswith("[AVERT") or "⚠" in s:
            return "avert"
        if s.startswith(("[ÉCHEC", "[ERREUR")) or "❌" in s or "💥" in s or s.startswith("Traceback"):
            return "mal"
        if s.startswith(("###", "====", "RAPPORT", "VERDICT", "BILAN")):
            return "titre"
        if s.startswith(("[SKIP", "---")):
            return "gris"
        return None

    def _ecrire_logs(self, contenu, en_bas=True):
        self.txt_logs.configure(state="normal")
        self.txt_logs.delete("1.0", "end")
        for ligne in contenu.splitlines():
            self.txt_logs.insert("end", T(ligne) + "\n", self._tag_ligne(ligne))
        self.txt_logs.configure(state="disabled")
        self.txt_logs.see("end" if en_bas else "1.0")

    def _sur_log(self, ligne):
        if not self.console_visible:
            return
        self.txt_logs.configure(state="normal")
        self.txt_logs.insert("end", T(ligne) + "\n", self._tag_ligne(ligne))
        self.txt_logs.configure(state="disabled")
        self.txt_logs.see("end")

    # ---------- page : outils ----------
    def _page_outils(self):
        page = ctk.CTkFrame(self.zone, fg_color="transparent")
        page.grid_columnconfigure((0, 1, 2), weight=1, uniform="outils")
        outils = [
            ("💤", "Dormir", "NREM (oubli + consolidation) + 2 rêves REM.",
             self.action_dormir),
            ("🌙", "Rêver", "Déclenche un cycle REM manuel (2 rêves, T dynamique).",
             self.action_rever),
            ("🧠", "Dissonances", "Voir l'état du démon de dissonance et son historique.",
             self.action_afficher_dissonances),
            ("🗜️", "Archives", "Notes compressées .cpk (plafond 1 Go).",
             lambda: self.afficher_page("archives")),
            ("📚", "Documents", "Documents écrits par Nyx (lecture à la demande).",
             lambda: self.afficher_page("documents")),
            ("📖", "Relire un doc", "Nyx relit spontanément un de ses documents.",
             self.action_lecture_spontanee),
            ("➕", "Nouvelle note", "Archive une nouvelle note compressée.",
             self.action_archiver_note),
            ("📤", "Exporter archives", "Exporte toutes les notes en .txt (décompressées).",
             self.action_exporter_archives),
            ("💭", "Penser à voix haute", "Nyx réfléchit seule à ses souvenirs qui s'estompent.",
             self.action_reflechir),
            ("🪞", "Identité", "Nyx se décrit à partir de son journal intime.",
             self.action_identite),
            ("🧹", "Nettoyer", "Repère et supprime les concepts pollués.",
             self.action_nettoyer),
            ("💾", "Purger la mémoire", "Archive les souvenirs faibles et en déclin.",
             self.action_purger_memoire),
            ("📊", "Taille mémoire", "Mesure l'espace disque utilisé.",
             self.action_verifier_taille),
            ("🔊", "Tester la voix", "Joue un court message pour tester.",
             self.action_tester_voix),
            ("⏹️", "Arrêter la voix", "Coupe la lecture immédiatement.",
             self.action_arreter_voix),
            ("🔄", "Recharger la voix", "Recharge Piper après un changement.",
             self.action_recharger_voix),
            ("🔎", "Check complet", "Auto-diagnostic complet avec le modèle.",
             lambda: self.action_check(False)),
            ("⚡", "Check rapide", "Auto-diagnostic sans appel au modèle.",
             lambda: self.action_check(True)),
            ("📂", "Dossier des logs", "Ouvre le dossier des rapports.",
             lambda: self.ouvrir_dossier("logs")),
        ]
        for i, (icone, titre, description, cmd) in enumerate(outils):
            carte = self._carte(page)
            carte.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")
            L(carte, icone, font=ctk.CTkFont(size=32)).pack(anchor="w", padx=20, pady=(16, 0))
            L(carte, titre, font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=20)
            L(carte, description, text_color=MUTED, wraplength=300, justify="left",
              anchor="w").pack(anchor="w", padx=20, pady=(2, 10))
            if titre == "Rêver":
                couleur_btn = VIOLET
                hover_btn = VIOLET_FONCE
            elif titre == "Dissonances":
                couleur_btn = INDIGO
                hover_btn = INDIGO_FONCE
            elif titre in ("Archives", "Documents", "Relire un doc", "Nouvelle note", "Exporter archives"):
                couleur_btn = TEAL
                hover_btn = TEAL_FONCE
            else:
                couleur_btn = ACCENT
                hover_btn = ACCENT_HOVER
            b = B(carte, "Lancer", width=110, height=32, fg_color=couleur_btn,
                  hover_color=hover_btn, command=cmd)
            b.pack(anchor="w", padx=20, pady=(0, 18))
            if titre not in ("Dossier des logs",):
                self.boutons_action.append(b)
        return page

    # ---------- page : réglages ----------
    def _page_reglages(self):
        page = ctk.CTkScrollableFrame(self.zone, fg_color="transparent")

        modeles = self._carte(page)
        modeles.pack(fill="x", pady=6)
        L(modeles, "Modèles Ollama", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 4))
        L(modeles, "Qwen2.5:3b est recommandé pour cohabiter avec Piper.",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        for titre, attribut, role in (("Conversation, journal, émotions", "opt_chat", "chat"),
                                      ("Consolidation du sommeil", "opt_conso", "consolidation")):
            ligne = ctk.CTkFrame(modeles, fg_color="transparent")
            ligne.pack(fill="x", padx=20, pady=4)
            L(ligne, titre, width=240, anchor="w", text_color=MUTED).pack(side="left")
            widget = self._option(ligne, ["…"], lambda v, r=role: self._changer_modele(r, v), 300)
            widget.pack(side="left")
            setattr(self, attribut, widget)
        B(modeles, "↻  Recharger la liste des modèles", fg_color=SURFACE2, hover_color=ACCENT,
          text_color=TEXT, command=self.charger_modeles).pack(anchor="w", padx=20, pady=(8, 16))

        emb = self._carte(page)
        emb.pack(fill="x", pady=6)
        L(emb, "Embeddings (mémoire sémantique)", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 4))
        L(emb, "Modèle Ollama qui transforme les textes en vecteurs.",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        ligne_emb = ctk.CTkFrame(emb, fg_color="transparent")
        ligne_emb.pack(fill="x", padx=20, pady=4)
        L(ligne_emb, "Modèle d'embedding", width=240, anchor="w", text_color=MUTED).pack(side="left")
        self.entree_embedding_model = ctk.CTkEntry(ligne_emb, width=300, fg_color=SURFACE2,
                                                   border_width=0, placeholder_text_color=MUTED)
        self.entree_embedding_model.pack(side="left")
        B(ligne_emb, "Appliquer", width=90, height=28, fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self._appliquer_embedding_model).pack(side="left", padx=8)
        self.lbl_embedding_etat = L(emb, "", text_color=MUTED, anchor="w")
        self.lbl_embedding_etat.pack(fill="x", padx=20, pady=(4, 16))

        lim = self._carte(page)
        lim.pack(fill="x", pady=6)
        L(lim, "Limite de mémoire sur disque", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 8))
        self.lbl_limite_go = L(lim, "", text_color=MUTED)
        self.lbl_limite_go.pack(anchor="w", padx=20, pady=(6, 0))
        self.sl_limite_go = ctk.CTkSlider(lim, from_=0.1, to=10.0, number_of_steps=99, width=360,
                                          button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                                          progress_color=ACCENT,
                                          command=lambda v: self._reglage_limite("memoire_limite_go", v))
        self.sl_limite_go.pack(anchor="w", padx=20, pady=(2, 8))
        self.sw_purge_auto = ctk.CTkSwitch(
            lim, text="Purger automatiquement les souvenirs faibles quand la limite est atteinte",
            progress_color=ACCENT,
            command=lambda: self._reglage_limite("memoire_purge_auto_actif", None))
        self.sw_purge_auto.pack(anchor="w", padx=20, pady=6)
        L(lim, "", height=8).pack()

        # ---------- ARCHIVES ----------
        arch = self._carte(page)
        arch.pack(fill="x", pady=6)
        L(arch, "🗜️  Archives compressées", font=ctk.CTkFont(size=15, weight="bold"),
          text_color=TEAL).pack(anchor="w", padx=20, pady=(16, 4))
        L(arch, "Notes compressées en .cpk, plafond 1 Go avec purge auto des plus anciennes. "
                "Dossier séparé de cerveau_memories/.",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        self.lbl_archives_limite = L(arch, "", text_color=MUTED)
        self.lbl_archives_limite.pack(anchor="w", padx=20, pady=(6, 0))
        self.sl_archives_limite = ctk.CTkSlider(
            arch, from_=0.1, to=10.0, number_of_steps=99, width=360,
            button_color=TEAL, button_hover_color=TEAL_FONCE,
            progress_color=TEAL,
            command=lambda v: self._reglage_archives_limite(v))
        self.sl_archives_limite.pack(anchor="w", padx=20, pady=(2, 8))
        self.sw_archives_compress = ctk.CTkSwitch(
            arch, text="Compresser automatiquement les notes ajoutées (.cpk)",
            progress_color=TEAL, command=self._reglage_archives_compress)
        self.sw_archives_compress.pack(anchor="w", padx=20, pady=6)
        L(arch, "", height=8).pack()

        # ---------- RÊVES ----------
        reves = self._carte(page)
        reves.pack(fill="x", pady=6)
        L(reves, "🌙  Rêves", font=ctk.CTkFont(size=15, weight="bold"),
          text_color=VIOLET).pack(anchor="w", padx=20, pady=(16, 4))
        L(reves, "Nyx rêve pendant son sommeil. 2 rêves avec température dynamique "
                 "(3 phases : 0.8 → 1.0 → 0.7 puis 0.9 → 1.1 → 0.8).",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        self.sw_reve_actif = ctk.CTkSwitch(
            reves, text="Activer les rêves (2 par sommeil)",
            progress_color=VIOLET, command=self._reglage_reve_actif)
        self.sw_reve_actif.pack(anchor="w", padx=20, pady=6)
        self.sw_reve_spontane = ctk.CTkSwitch(
            reves, text="Nyx peut mentionner ses rêves spontanément",
            progress_color=VIOLET, command=self._reglage_reve_spontane)
        self.sw_reve_spontane.pack(anchor="w", padx=20, pady=6)
        ligne_test_reve = ctk.CTkFrame(reves, fg_color="transparent")
        ligne_test_reve.pack(fill="x", padx=20, pady=(12, 16))
        b_rever = B(ligne_test_reve, "🌙  Rêver maintenant (Ctrl+R)", fg_color=VIOLET,
                    hover_color=VIOLET_FONCE, text_color="#ffffff", command=self.action_rever)
        b_rever.pack(side="left")
        self.boutons_action.append(b_rever)
        B(ligne_test_reve, "📜  Voir les rêves", fg_color=SURFACE2, hover_color=VIOLET,
          text_color=TEXT, command=lambda: self.afficher_page("reves")).pack(side="left", padx=8)

        # ---------- DISSONANCE ----------
        diss = self._carte(page)
        diss.pack(fill="x", pady=6)
        L(diss, "🧠  Démon de Dissonance", font=ctk.CTkFont(size=15, weight="bold"),
          text_color=INDIGO).pack(anchor="w", padx=20, pady=(16, 4))
        L(diss, "Détecte les conflits internes (croyances contradictoires) et les résout. "
                "Mode simulation par défaut : observe et log sans agir.",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        self.sw_diss_simulation = ctk.CTkSwitch(
            diss, text="Mode simulation (recommandé : observer d'abord)",
            progress_color=INDIGO, command=self._reglage_diss_simulation)
        self.sw_diss_simulation.pack(anchor="w", padx=20, pady=6)
        self.lbl_diss_seuil = L(diss, "", text_color=MUTED)
        self.lbl_diss_seuil.pack(anchor="w", padx=20, pady=(10, 0))
        self.sl_diss_seuil = ctk.CTkSlider(
            diss, from_=0.1, to=1.0, number_of_steps=18, width=360,
            button_color=INDIGO, button_hover_color=INDIGO_FONCE,
            progress_color=INDIGO,
            command=lambda v: self._reglage_diss_seuil(v))
        self.sl_diss_seuil.pack(anchor="w", padx=20, pady=(2, 6))
        self.lbl_diss_budget = L(diss, "", text_color=MUTED)
        self.lbl_diss_budget.pack(anchor="w", padx=20, pady=(10, 0))
        self.sl_diss_budget = ctk.CTkSlider(
            diss, from_=0, to=20, number_of_steps=20, width=360,
            button_color=INDIGO, button_hover_color=INDIGO_FONCE,
            progress_color=INDIGO,
            command=lambda v: self._reglage_diss_budget(v))
        self.sl_diss_budget.pack(anchor="w", padx=20, pady=(2, 16))

        # ---------- VOIX ----------
        voix = self._carte(page)
        voix.pack(fill="x", pady=6)
        L(voix, "🗣️  Voix (Piper TTS)", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 4))
        L(voix, "Nyx parle via Piper (fr_FR-siwis-medium). Léger, rapide, sans GPU.",
          text_color=MUTED, wraplength=800, justify="left").pack(anchor="w", padx=20, pady=(0, 8))

        self.sw_voix = ctk.CTkSwitch(
            voix, text="Activer la voix (Nyx parle à voix haute)",
            progress_color=ACCENT, command=self._reglage_voix_actif)
        self.sw_voix.pack(anchor="w", padx=20, pady=6)
        self.sw_pensees = ctk.CTkSwitch(
            voix, text="Prononcer les pensées spontanées à voix haute",
            progress_color=ACCENT, command=self._reglage_pensees_actif)
        self.sw_pensees.pack(anchor="w", padx=20, pady=6)
        self.sw_interpeller = ctk.CTkSwitch(
            voix, text="Nyx peut t'interpeller spontanément",
            progress_color=ACCENT, command=self._reglage_interpeller)
        self.sw_interpeller.pack(anchor="w", padx=20, pady=6)

        ligne_test_voix = ctk.CTkFrame(voix, fg_color="transparent")
        ligne_test_voix.pack(fill="x", padx=20, pady=(8, 4))
        B(ligne_test_voix, "▶️  Tester la voix", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=self.action_tester_voix).pack(side="left")
        B(ligne_test_voix, "⏹️  Arrêter", fg_color=SURFACE2, hover_color=BAD, text_color=TEXT,
          command=self.action_arreter_voix).pack(side="left", padx=8)
        B(ligne_test_voix, "📊  État", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=self.action_etat_voix).pack(side="left", padx=8)

        self.curseurs_voix = {}
        for cle, titre, mini, maxi, pas in (
            ("voix_longueur", "Vitesse (>1 = plus lent)", 0.5, 2.0, 0.05),
            ("voix_volume", "Volume", 0.0, 2.0, 0.1),
        ):
            etiquette = L(voix, titre, text_color=MUTED)
            etiquette.pack(anchor="w", padx=20, pady=(10, 0))
            curseur = ctk.CTkSlider(
                voix, from_=mini, to=maxi, number_of_steps=int((maxi - mini) / pas),
                button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                width=360, progress_color=ACCENT,
                command=lambda v, c=cle: self._reglage_voix_seuil(c, v))
            curseur.pack(anchor="w", padx=20, pady=(2, 0))
            self.curseurs_voix[cle] = (curseur, etiquette, titre)

        self.lbl_voix_etat_reglages = L(voix, "", text_color=MUTED, anchor="w",
                                        font=ctk.CTkFont(size=11))
        self.lbl_voix_etat_reglages.pack(fill="x", padx=20, pady=(4, 16))

        # ---------- COMPORTEMENT ----------
        comportement = self._carte(page)
        comportement.pack(fill="x", pady=6)
        L(comportement, "Comportement", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 8))
        self.sw_prefrontal = ctk.CTkSwitch(
            comportement, text="Réflexion préfrontale (plan interne avant chaque réponse)",
            progress_color=ACCENT, command=lambda: self._reglage("prefrontal"))
        self.sw_prefrontal.pack(anchor="w", padx=20, pady=6)
        self.sw_debug = ctk.CTkSwitch(
            comportement, text="Mode debug : afficher les distances de rappel dans la console",
            progress_color=ACCENT, command=lambda: self._reglage("debug"))
        self.sw_debug.pack(anchor="w", padx=20, pady=6)
        self.lbl_distance = L(comportement, "", text_color=MUTED)
        self.lbl_distance.pack(anchor="w", padx=20, pady=(10, 0))
        self.sl_distance = ctk.CTkSlider(comportement, from_=0.5, to=2.0, number_of_steps=30, width=360,
                                         button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                                         progress_color=ACCENT,
                                         command=lambda v: self._reglage("distance", v))
        self.sl_distance.pack(anchor="w", padx=20, pady=(2, 18))

        # ---------- PULSIONS ----------
        pulsions = self._carte(page)
        pulsions.pack(fill="x", pady=6)
        L(pulsions, "Pulsions : sommeil, pensées et lectures spontanées",
          font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=20, pady=(16, 8))
        self.sw_sommeil_auto = ctk.CTkSwitch(
            pulsions, text="Sommeil automatique",
            progress_color=ACCENT, command=lambda: self._reglage("sommeil_auto"))
        self.sw_sommeil_auto.pack(anchor="w", padx=20, pady=6)
        self.sw_reflexion_auto = ctk.CTkSwitch(
            pulsions, text="Pensées spontanées (à voix haute)",
            progress_color=ACCENT, command=lambda: self._reglage("reflexion_auto"))
        self.sw_reflexion_auto.pack(anchor="w", padx=20, pady=6)
        self.sw_lecture_auto = ctk.CTkSwitch(
            pulsions, text="Lectures spontanées (relire un de ses documents)",
            progress_color=ACCENT, command=lambda: self._reglage("lecture_auto"))
        self.sw_lecture_auto.pack(anchor="w", padx=20, pady=6)
        self.curseurs = {}
        for cle, titre, mini, maxi in (
            ("SEUIL_ENVIE_DORMIR", "Seuil « a sommeil »", 0.2, 0.95),
            ("SEUIL_SOMMEIL_AUTO", "Seuil de sommeil automatique", 0.3, 1.0),
            ("SEUIL_REFLEXION", "Seuil d'envie de penser à voix haute", 0.2, 0.95),
            ("SEUIL_ENVIE_LIRE", "Seuil d'envie de lire un document", 0.2, 0.95),
            ("POIDS_HUMEUR", "Poids de l'humeur sur le rappel", 0.0, 0.6),
        ):
            etiquette = L(pulsions, titre, text_color=MUTED)
            etiquette.pack(anchor="w", padx=20, pady=(10, 0))
            curseur = ctk.CTkSlider(pulsions, from_=mini, to=maxi,
                                    number_of_steps=int(round((maxi - mini) / 0.05)),
                                    button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                                    width=360, progress_color=ACCENT,
                                    command=lambda v, c=cle: self._reglage_seuil(c, v))
            curseur.pack(anchor="w", padx=20, pady=(2, 0))
            self.curseurs[cle] = (curseur, etiquette, titre)
        L(pulsions, "", height=8).pack()

        # ---------- APPARENCE ----------
        apparence = self._carte(page)
        apparence.pack(fill="x", pady=6)
        L(apparence, "Apparence", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(16, 8))
        self.seg_theme = ctk.CTkSegmentedButton(
            apparence, values=["Sombre", "Clair", "Système"],
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            command=lambda v: self.changer_apparence(
                {"Sombre": "dark", "Clair": "light", "Système": "system"}[v]))
        self.seg_theme.pack(anchor="w", padx=20, pady=(0, 10))
        self.seg_theme.set({"dark": "Sombre", "light": "Clair", "system": "Système"}.get(
            REGLAGES.get("apparence", "dark"), "Sombre"))
        self.sw_emoji = ctk.CTkSwitch(
            apparence,
            text="Icônes emoji (redémarrage requis ; désactivez-les si l'affichage pose problème)",
            progress_color=ACCENT, command=lambda: self._reglage("emoji"))
        self.sw_emoji.pack(anchor="w", padx=20, pady=(0, 18))
        if not SANS_EMOJI:
            self.sw_emoji.select()
        return page

    # ---------- méthodes de chargement/réglages ----------
    def charger_modeles(self):
        if not self.agent:
            return
        self.run_async(self.agent._lister_modeles_ollama, self._apres_modeles, exclusif=False)

    def _apres_modeles(self, noms, _sortie):
        noms = noms or [self.agent.model_name]
        for opt, courant in ((self.opt_chat, self.agent.model_name),
                             (self.opt_conso, self.agent.model_consolidation)):
            valeurs = noms if courant in noms else [courant] + noms
            opt.configure(values=valeurs)
            opt.set(courant)
        self.entree_embedding_model.delete(0, "end")
        self.entree_embedding_model.insert(0, self.agent.embedding_model)
        if getattr(self.agent, "ef", None) is not None:
            self.lbl_embedding_etat.configure(
                text=T(f"✅ Embedding actif : {self.agent.embedding_model} (768 dims)"),
                text_color=GOOD)
        else:
            self.lbl_embedding_etat.configure(
                text=T("⚠️  Embedding nomic inactif"),
                text_color=WARN)
        info = self.agent.verifier_voix()
        if info.disponible:
            self.lbl_voix_etat_reglages.configure(
                text=T(f"✅ {info.message}"),
                text_color=GOOD)
        else:
            self.lbl_voix_etat_reglages.configure(
                text=T(f"❌ {info.message}"),
                text_color=BAD)

    def _changer_modele(self, role, nom):
        if role == "chat":
            self.agent.model_name = nom
            REGLAGES["modele_chat"] = nom
            self._pastille(nom, GOOD)
        else:
            self.agent.model_consolidation = nom
            REGLAGES["modele_consolidation"] = nom
        self.sauver_reglages()
        self.toast(f"Modèle {role} : {nom}", GOOD)

    def _appliquer_embedding_model(self):
        nouveau = self.entree_embedding_model.get().strip()
        if not nouveau:
            self.toast("Le nom du modèle ne peut pas être vide.", WARN)
            return
        self.agent.cfg.definir("embedding_model", nouveau)
        REGLAGES["embedding_model"] = nouveau
        self.sauver_reglages()
        self.toast(f"Modèle d'embedding enregistré : {nouveau}. Redémarrage requis.", WARN)

    def _reglage(self, cle, valeur=None):
        a = self.agent
        if cle == "prefrontal" and a:
            a.reflexion_prefrontale = bool(self.sw_prefrontal.get())
        elif cle == "debug" and a:
            a.debug = bool(self.sw_debug.get())
        elif cle == "distance" and a:
            a.SEUIL_DISTANCE = round(float(valeur), 2)
            self.lbl_distance.configure(
                text=f"Seuil de distance du rappel : {a.SEUIL_DISTANCE}")
        elif cle == "sommeil_auto" and a:
            a.sommeil_auto_actif = bool(self.sw_sommeil_auto.get())
            REGLAGES["sommeil_auto"] = a.sommeil_auto_actif
            self.sauver_reglages()
        elif cle == "reflexion_auto" and a:
            a.reflexion_auto_actif = bool(self.sw_reflexion_auto.get())
            REGLAGES["reflexion_auto"] = a.reflexion_auto_actif
            self.sauver_reglages()
        elif cle == "lecture_auto" and a:
            a.lecture_auto_actif = bool(self.sw_lecture_auto.get())
            REGLAGES["lecture_auto"] = a.lecture_auto_actif
            self.sauver_reglages()
        elif cle == "emoji":
            REGLAGES["sans_emoji"] = not bool(self.sw_emoji.get())
            self.sauver_reglages()
            self.toast("Réglage enregistré : redémarrez l'interface pour l'appliquer.", WARN)

    def _reglage_seuil(self, cle, valeur):
        if not self.agent:
            return
        valeur = round(float(valeur), 2)
        setattr(self.agent, cle, valeur)
        REGLAGES[cle.lower()] = valeur
        curseur, etiquette, titre = self.curseurs[cle]
        etiquette.configure(text=T(f"{titre} : {valeur:.2f}"))
        self.sauver_reglages()
        self.maj_pulsions()

    def _reglage_limite(self, cle, valeur):
        if not self.agent:
            return
        if cle == "memoire_limite_go" and valeur is not None:
            v = round(float(valeur), 2)
            if v < 0.1:
                v = 0.1
            self.agent.cfg.definir("memoire_limite_go", v)
            REGLAGES["memoire_limite_go"] = v
            self.lbl_limite_go.configure(text=T(f"Limite actuelle : {v:.2f} Go"))
            self.maj_taille_memoire()
        elif cle == "memoire_purge_auto_actif":
            v = bool(self.sw_purge_auto.get())
            self.agent.cfg.definir("memoire_purge_auto_actif", v)
            REGLAGES["memoire_purge_auto_actif"] = v
        self.sauver_reglages()

    def _reglage_archives_limite(self, valeur):
        if not self.agent:
            return
        v = round(float(valeur), 2)
        if v < 0.1:
            v = 0.1
        self.agent.cfg.definir("archives_limite_go", v)
        REGLAGES["archives_limite_go"] = v
        self.lbl_archives_limite.configure(text=T(f"Plafond archives : {v:.2f} Go"))
        self.maj_taille_memoire()
        self.sauver_reglages()

    def _reglage_archives_compress(self):
        if not self.agent:
            return
        v = bool(self.sw_archives_compress.get())
        self.agent.cfg.definir("archives_compresser_auto", v)
        REGLAGES["archives_compresser_auto"] = v
        self.sauver_reglages()

    def _reglage_voix_actif(self):
        if not self.agent:
            return
        actif = bool(self.sw_voix.get())
        self.agent.basculer_voix(actif)
        REGLAGES["voix_actif"] = actif
        self.sauver_reglages()
        self._maj_bouton_voix()

    def _reglage_pensees_actif(self):
        if not self.agent:
            return
        actif = bool(self.sw_pensees.get())
        self.agent.basculer_pensees(actif)
        REGLAGES["voix_pensees"] = actif
        self.sauver_reglages()
        self._maj_bouton_pensees()

    def _reglage_interpeller(self):
        if not self.agent:
            return
        actif = bool(self.sw_interpeller.get())
        self.agent.cfg.definir("voix_interpeller", actif)
        REGLAGES["voix_interpeller"] = actif
        self.sauver_reglages()

    def _reglage_voix_seuil(self, cle, valeur):
        if not self.agent:
            return
        v = round(float(valeur), 2)
        self.agent.cfg.definir(cle, v)
        REGLAGES[cle] = v
        curseur, etiquette, titre = self.curseurs_voix[cle]
        etiquette.configure(text=T(f"{titre} : {v:.2f}"))
        if self.agent.voix is not None:
            if cle == "voix_longueur":
                self.agent.voix.longueur_scale = v
            elif cle == "voix_volume":
                self.agent.voix.volume = v
        self.sauver_reglages()

    def _reglage_reve_actif(self):
        if not self.agent:
            return
        actif = bool(self.sw_reve_actif.get())
        self.agent.cfg.definir("reve_actif", actif)
        REGLAGES["reve_actif"] = actif
        self.sauver_reglages()
        self.toast(f"🌙 Rêves {'activés' if actif else 'désactivés'}.", VIOLET if actif else WARN)

    def _reglage_reve_spontane(self):
        if not self.agent:
            return
        actif = bool(self.sw_reve_spontane.get())
        self.agent.cfg.definir("reve_raconter_spontanement", actif)
        REGLAGES["reve_raconter_spontanement"] = actif
        self.sauver_reglages()

    def _reglage_diss_simulation(self):
        if not self.agent or self.agent.demon is None:
            return
        nouveau = bool(self.sw_diss_simulation.get())
        self.agent.demon.etat.mode_simulation = nouveau
        self.agent.cfg.definir("dissonance_mode_simulation", nouveau)
        REGLAGES["dissonance_mode_simulation"] = nouveau
        self.sauver_reglages()
        self.toast(f"🧠 Simulation : {'ON' if nouveau else 'OFF (actions réelles)'}",
                   WARN if nouveau else GOOD)

    def _reglage_diss_seuil(self, valeur):
        if not self.agent or self.agent.demon is None:
            return
        v = round(float(valeur), 2)
        self.agent.demon.seuil = v
        self.agent.demon.etat.seuil = v
        self.agent.cfg.definir("dissonance_seuil", v)
        REGLAGES["dissonance_seuil"] = v
        self.lbl_diss_seuil.configure(text=T(f"Seuil d'activation : {v:.2f}"))
        self.sauver_reglages()

    def _reglage_diss_budget(self, valeur):
        if not self.agent or self.agent.demon is None:
            return
        v = int(valeur)
        self.agent.demon.budget_heure = v
        self.agent.cfg.definir("dissonance_budget_heure", v)
        REGLAGES["dissonance_budget_heure"] = v
        self.lbl_diss_budget.configure(text=T(f"Budget d'actions : {v}/heure"))
        self.sauver_reglages()

    def changer_apparence(self, mode):
        ctk.set_appearance_mode(mode)
        REGLAGES["apparence"] = mode
        self.sauver_reglages()
        self._style_arbre()
        self.dessiner_histo()
        self.rendre_memoire()

    def sauver_reglages(self):
        try:
            with open(FICHIER_REGLAGES, "w", encoding="utf-8") as f:
                json.dump(REGLAGES, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---------- actions ----------
    def action_dormir(self):
        if not self.agent:
            return
        if not messagebox.askyesno("Dormir",
                                   "Nyx va oublier ses souvenirs faibles, consolider les neutres "
                                   "en concepts, et faire 2 rêves REM.\n\nContinuer ?"):
            return

        def apres(resultat, sortie_console):
            contenu = sortie_console or ""
            if resultat:
                contenu = (contenu + "\n\n" if contenu else "") + f"Résumé : {resultat}"
            self.fenetre_texte("💤 Compte rendu du sommeil", contenu)
            self.maj_emotions()
            self.rafraichir()

        self.run_async(self.agent.dormir, apres, "Nyx dort (NREM + REM)…")

    def action_identite(self):
        if not self.agent:
            return

        def apres(resultat, sortie_console):
            contenu = resultat or sortie_console or "(aucune réponse)"
            self.fenetre_texte("🪞 Identité de Nyx", contenu)

        self.run_async(self.agent.se_decrire, apres, "Introspection…")

    def action_reflechir(self):
        if not self.agent:
            return

        def apres(pensee, sortie_console):
            if not pensee:
                self.toast("⚠️ Aucune pensée produite.", WARN)
            self.maj_emotions()
            self.rafraichir()

        self.run_async(lambda: self.agent.auto_reflexion(force=True), apres,
                       "Nyx réfléchit à voix haute…")

    def action_purger_memoire(self):
        if not self.agent:
            return
        stats = self.stats_taille
        if stats and not stats.au_dessus_avertissement:
            if not messagebox.askyesno(
                    "Purger",
                    f"La mémoire est à {stats.fraction * 100:.1f} % de sa limite.\n\n"
                    "Purger quand même ?"):
                return
        else:
            if not messagebox.askyesno(
                    "Purger la mémoire",
                    "Cette action archive les souvenirs faibles et en déclin.\n\nContinuer ?"):
                return
        self.run_async(self.agent.purger_memoire,
                       lambda r, s: (self.fenetre_texte("💾 Résultat de la purge", self._resume_purge(r)),
                                     self.maj_taille_memoire(), self.rafraichir()),
                       "Purge de la mémoire…")

    @staticmethod
    def _resume_purge(resultat):
        if not resultat or not isinstance(resultat, dict):
            return "Aucune purge effectuée."
        if resultat.get("raison") == "sous le seuil d'avertissement":
            return "Sous le seuil d'avertissement : rien à purger."
        if resultat.get("raison") == "purge auto désactivée":
            return "Purge automatique désactivée."
        if resultat.get("raison") == "aucun candidat":
            return "Aucun souvenir faible ou en déclin à purger."
        avant = resultat.get("stats_avant")
        apres = resultat.get("stats_apres")
        lignes = ["Purge effectuée :", ""]
        if avant and apres:
            lignes.append(f"  Avant : {avant.resume()}")
            lignes.append(f"  Après : {apres.resume()}")
        return "\n".join(lignes)

    def action_verifier_taille(self):
        if not self.agent:
            return
        try:
            stats = self.agent.verifier_taille_memoire()
            self.stats_taille = stats
            self.maj_taille_memoire()
            self.toast(f"💾 {stats.resume()}", couleur_taille(stats))
        except Exception as e:
            self.toast(f"Impossible de vérifier : {e}", BAD)

    def action_nettoyer(self):
        if not self.agent:
            return

        def chercher():
            a = self.agent
            if a.hippocampe.count() == 0:
                return []
            data = a.hippocampe.get()
            return [(i, d) for i, d, m in zip(data['ids'], data['documents'], data['metadatas'])
                    if m and a._normaliser_meta(m)['type'] == 'concept_consolidé'
                    and not a._concept_valide(d)]

        self.run_async(chercher, lambda suspects, s: self._dialogue_nettoyage(suspects),
                       "Analyse des concepts…")

    def _dialogue_nettoyage(self, suspects):
        if not suspects:
            self.toast("✅ Aucun concept suspect.", GOOD)
            return
        fen = ctk.CTkToplevel(self)
        fen.title("Concepts suspects")
        fen.geometry("780x480")
        fen.transient(self)
        L(fen, f"🧹  {len(suspects)} concept(s) suspect(s)",
          font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(18, 4))
        boite = ctk.CTkTextbox(fen, wrap="word")
        boite.pack(fill="both", expand=True, padx=20, pady=12)
        boite.insert("end", "\n\n".join("• " + T(d) for _, d in suspects))
        boite.configure(state="disabled")
        rangee = ctk.CTkFrame(fen, fg_color="transparent")
        rangee.pack(pady=(0, 16))

        def supprimer():
            fen.destroy()
            ids = [i for i, _ in suspects]
            self.run_async(lambda: self.agent.hippocampe.delete(ids=ids),
                           lambda r, s: (self.toast(f"🧹 {len(ids)} concept(s) supprimé(s).", GOOD),
                                         self.rafraichir()), "Nettoyage…")

        B(rangee, "🗑️  Supprimer", fg_color=BAD, hover_color="#d94444",
          command=supprimer).pack(side="left", padx=8)
        B(rangee, "Annuler", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=fen.destroy).pack(side="left", padx=8)
        fen.after(200, fen.focus)

    def action_check(self, rapide):
        if not self.agent:
            return

        def apres(chemin, _sortie):
            self.afficher_page("logs")
            self.charger_liste_logs()
            if chemin:
                nom = os.path.basename(chemin)
                self.selection_logs.set(nom)
                self._choisir_log(nom)
            self.toast("🔎 Diagnostic terminé.", GOOD)

        self.run_async(lambda: self.agent.check(rapide=rapide), apres,
                       "Check rapide…" if rapide else "Check complet (≈ 30 s)…")

    def fenetre_texte(self, titre, texte):
        fen = ctk.CTkToplevel(self)
        fen.title(T(titre))
        fen.geometry("760x520")
        fen.transient(self)
        boite = ctk.CTkTextbox(fen, wrap="word", font=ctk.CTkFont(size=13))
        boite.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        boite.insert("end", T(texte) or "(aucune sortie)")
        boite.configure(state="disabled")
        rangee = ctk.CTkFrame(fen, fg_color="transparent")
        rangee.pack(pady=(0, 14))
        B(rangee, "📋  Copier", fg_color=SURFACE2, hover_color=ACCENT, text_color=TEXT,
          command=lambda: self._copier(texte)).pack(side="left", padx=6)
        B(rangee, "Fermer", fg_color=ACCENT, hover_color=ACCENT_HOVER,
          command=fen.destroy).pack(side="left", padx=6)
        fen.after(200, fen.focus)

    def afficher_aide(self):
        lignes = ["FONCTIONS DE L'INTERFACE", "",
                  "• Conversation : Entrée = envoyer, Maj+Entrée = nouvelle ligne.",
                  "• Texte sélectionnable : clique dans une bulle, sélectionne, Ctrl+C.",
                  "• Ctrl+A dans une bulle : sélectionne tout. Clic droit : copier.",
                  "• 🌙 Rêver (Ctrl+R) : déclenche 2 rêves REM.",
                  "• 🧠 Dissonances (Ctrl+D) : voir l'état du démon de dissonance.",
                  "• 🗜️ Archives (Ctrl+A) : notes compressées .cpk.",
                  "• 📚 Documents (Ctrl+L) : documents écrits par Nyx.",
                  "• Mémoire : filtrer, trier, rechercher et supprimer des souvenirs.",
                  "• Journal : lire, filtrer, ajouter une réflexion, se décrire.",
                  "• État & émotions : jauges VAD, statistiques, mémoire, voix, rêves, "
                  "dissonances, archives, documents.",
                  "• Logs & diagnostic : console en direct, rapports de check.",
                  "• Outils : dormir, rêver, dissonances, archives, documents, penser, "
                  "relire un doc, identité, nettoyer, purger, voix, check.",
                  "• Réglages : modèles, embeddings, mémoire, archives, rêves, dissonances, "
                  "voix, pulsions, apparence.", "",
                  "DOCUMENTS & LECTURE", "",
                  "• « écris-moi un document sur… » → Nyx écrit un .txt.",
                  "• « lis ton document sur… » ou « qu'as-tu écrit sur… » → Nyx relit.",
                  "• « quels documents as-tu écrits ? » → Nyx liste ses documents.",
                  "• 📖 Relire un document (bouton ou Ctrl+L) → lecture spontanée.",
                  "• Jauge « Envie de lire » : monte avec le temps, déclenche des "
                  "lectures spontanées quand Nyx est inactive.", "",
                  "RÊVES", "",
                  "• 2 rêves par sommeil avec température dynamique (3 phases).",
                  "• Rêve 1 : 0.8 → 1.0 → 0.7. Rêve 2 : 0.9 → 1.1 → 0.8.",
                  "• Les rêves marquants deviennent des souvenirs oniriques (🌙).",
                  "• Avant de rêver : relecture d'une note d'archive.",
                  "• Après : archivage auto des rêves marquants.", "",
                  "DISSONANCES", "",
                  "• Détecte les conflits internes (croyances contradictoires).",
                  "• Choisit le levier de résolution (changer A, changer B, synthétiser, réduire).",
                  "• Mode SIMULATION par défaut : observe et log sans agir.",
                  "• Garde-fous : seuil, budget horaire, intervalle min, pause.", "",
                  "ARCHIVES", "",
                  "• Notes compressées en .cpk (store/zlib/bz2/lzma/lzma_delta).",
                  "• Classification automatique par extension.",
                  "• Plafond 1 Go (configurable), purge auto des plus anciennes.",
                  "• Dossier séparé de cerveau_memories/.",
                  "• Intention dans le chat : « Archive ça » archive la dernière réponse.", "",
                  "VOIX (Piper TTS)", "",
                  "• Nyx parle via Piper (Siwis FR), léger, rapide, sans GPU.",
                  "• Espace = arrêt, Ctrl+M = mute, Ctrl+P = pensées.", "",
                  "ÉQUIVALENTS EN LIGNE DE COMMANDE", ""]
        lignes += [f"  {cmd:<22}{desc}" for cmd, desc in AgentCerebralNyx.COMMANDES_AIDE]
        self.fenetre_texte("📖 Aide", "\n".join(lignes))

    def ouvrir_dossier(self, chemin):
        chemin = os.path.abspath(chemin)
        os.makedirs(chemin, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(chemin)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", chemin])
            else:
                subprocess.Popen(["xdg-open", chemin])
        except Exception as e:
            self.toast(f"Impossible d'ouvrir le dossier : {e}", BAD)

    # ---------- démarrage / arrêt ----------
    def _init_agent(self):
        self._afficher_overlay(
            "Chargement de Nyx…",
            "Mémoire, embeddings, voix Piper, rêves, dissonance, archives, documents…"
        )

        kw = {}
        if REGLAGES.get("modele_chat"):
            kw["model_name"] = REGLAGES["modele_chat"]
        if REGLAGES.get("modele_consolidation"):
            kw["model_consolidation"] = REGLAGES["modele_consolidation"]

        def apres(agent, _sortie):
            self.agent = agent

            for cle in ("SEUIL_ENVIE_DORMIR", "SEUIL_SOMMEIL_AUTO", "SEUIL_REFLEXION",
                        "SEUIL_ENVIE_LIRE", "POIDS_HUMEUR"):
                if cle.lower() in REGLAGES:
                    setattr(agent, cle, REGLAGES[cle.lower()])
                if cle in self.curseurs:
                    curseur, etiquette, titre = self.curseurs[cle]
                    curseur.set(getattr(agent, cle))
                    etiquette.configure(text=T(f"{titre} : {getattr(agent, cle):.2f}"))

            if "memoire_limite_go" in REGLAGES:
                agent.cfg.definir("memoire_limite_go", REGLAGES["memoire_limite_go"])
            if "memoire_purge_auto_actif" in REGLAGES:
                agent.cfg.definir("memoire_purge_auto_actif", REGLAGES["memoire_purge_auto_actif"])
            self.sl_limite_go.set(agent.memoire_limite_go)
            self.lbl_limite_go.configure(text=T(f"Limite actuelle : {agent.memoire_limite_go:.2f} Go"))
            if agent.memoire_purge_auto_actif:
                self.sw_purge_auto.select()
            else:
                self.sw_purge_auto.deselect()

            # Archives
            if "archives_limite_go" in REGLAGES:
                agent.cfg.definir("archives_limite_go", REGLAGES["archives_limite_go"])
            if "archives_compresser_auto" in REGLAGES:
                agent.cfg.definir("archives_compresser_auto", REGLAGES["archives_compresser_auto"])
            self.sl_archives_limite.set(agent.cfg.archives_limite_go)
            self.lbl_archives_limite.configure(
                text=T(f"Plafond archives : {agent.cfg.archives_limite_go:.2f} Go"))
            if agent.cfg.archives_compresser_auto:
                self.sw_archives_compress.select()
            else:
                self.sw_archives_compress.deselect()

            self.entree_embedding_model.delete(0, "end")
            self.entree_embedding_model.insert(0, agent.embedding_model)
            if getattr(agent, "ef", None) is not None:
                self.lbl_embedding_etat.configure(
                    text=T(f"✅ Embedding actif : {agent.embedding_model} (768 dims)"),
                    text_color=GOOD)
            else:
                self.lbl_embedding_etat.configure(
                    text=T("⚠️  Embedding nomic inactif"),
                    text_color=WARN)

            if "reve_actif" in REGLAGES:
                agent.cfg.definir("reve_actif", REGLAGES["reve_actif"])
            if "reve_raconter_spontanement" in REGLAGES:
                agent.cfg.definir("reve_raconter_spontanement", REGLAGES["reve_raconter_spontanement"])
            self.sw_reve_actif.select() if agent.cfg.reve_actif else self.sw_reve_actif.deselect()
            self.sw_reve_spontane.select() if agent.cfg.reve_raconter_spontanement else self.sw_reve_spontane.deselect()

            # Dissonance
            if agent.demon is not None:
                if "dissonance_mode_simulation" in REGLAGES:
                    agent.demon.etat.mode_simulation = REGLAGES["dissonance_mode_simulation"]
                    agent.cfg.definir("dissonance_mode_simulation",
                                     REGLAGES["dissonance_mode_simulation"])
                if "dissonance_seuil" in REGLAGES:
                    v = REGLAGES["dissonance_seuil"]
                    agent.demon.seuil = v
                    agent.demon.etat.seuil = v
                    agent.cfg.definir("dissonance_seuil", v)
                if "dissonance_budget_heure" in REGLAGES:
                    v = REGLAGES["dissonance_budget_heure"]
                    agent.demon.budget_heure = v
                    agent.cfg.definir("dissonance_budget_heure", v)
                self.sw_diss_simulation.select() if agent.demon.etat.mode_simulation else self.sw_diss_simulation.deselect()
                self.sl_diss_seuil.set(agent.demon.seuil)
                self.lbl_diss_seuil.configure(text=T(f"Seuil d'activation : {agent.demon.seuil:.2f}"))
                self.sl_diss_budget.set(agent.demon.budget_heure)
                self.lbl_diss_budget.configure(text=T(f"Budget d'actions : {agent.demon.budget_heure}/heure"))

            # Voix
            if "voix_actif" in REGLAGES:
                agent.cfg.definir("voix_actif", REGLAGES["voix_actif"])
            if "voix_pensees" in REGLAGES:
                agent.cfg.definir("voix_pensees", REGLAGES["voix_pensees"])
            if "voix_interpeller" in REGLAGES:
                agent.cfg.definir("voix_interpeller", REGLAGES["voix_interpeller"])
            self.sw_voix.select() if agent.cfg.voix_actif else self.sw_voix.deselect()
            self.sw_pensees.select() if agent.cfg.voix_pensees else self.sw_pensees.deselect()
            self.sw_interpeller.select() if agent.cfg.voix_interpeller else self.sw_interpeller.deselect()
            for cle, (curseur, etiquette, titre) in self.curseurs_voix.items():
                v = getattr(agent.cfg, cle)
                curseur.set(v)
                etiquette.configure(text=T(f"{titre} : {v:.2f}"))

            info_v = agent.verifier_voix()
            if info_v.disponible:
                self.lbl_voix_etat_reglages.configure(
                    text=T(f"✅ {info_v.message}"),
                    text_color=GOOD)
            else:
                self.lbl_voix_etat_reglages.configure(
                    text=T(f"❌ {info_v.message}"),
                    text_color=BAD)
            self._maj_bouton_voix()
            self._maj_bouton_pensees()

            agent.sommeil_auto_actif = REGLAGES.get("sommeil_auto", True)
            agent.reflexion_auto_actif = REGLAGES.get("reflexion_auto", True)
            agent.lecture_auto_actif = REGLAGES.get("lecture_auto", True)
            self.sw_sommeil_auto.select() if agent.sommeil_auto_actif else self.sw_sommeil_auto.deselect()
            self.sw_reflexion_auto.select() if agent.reflexion_auto_actif else self.sw_reflexion_auto.deselect()
            self.sw_lecture_auto.select() if agent.lecture_auto_actif else self.sw_lecture_auto.deselect()
            agent.sur_evenement = self._evenement_agent
            agent.demarrer_pulsions(5.0)

            self.sw_prefrontal.select() if agent.reflexion_prefrontale else self.sw_prefrontal.deselect()
            self.sl_distance.set(agent.SEUIL_DISTANCE)
            self._reglage("distance", agent.SEUIL_DISTANCE)

            self._retirer_overlay()

            message = (f"Bonjour {agent.utilisateur}. Ma mémoire est chargée : "
                       f"{agent.hippocampe.count()} souvenir(s), {agent.journal_intime.count()} "
                       f"entrée(s) de journal.")
            try:
                nb_archives = agent.stats_archives()["total_notes"]
                if nb_archives:
                    message += f" {nb_archives} note(s) archivée(s)."
            except Exception:
                pass
            try:
                nb_docs = len(agent.lister_documents())
                if nb_docs:
                    message += f" Et {nb_docs} document(s) écrit(s)."
            except Exception:
                pass
            message += " De quoi voulons-nous parler ?"
            if agent.cfg.reve_actif:
                message += " (Je rêve aussi, tu peux me demander mes rêves.)"
            self.ajouter_bulle("nyx", message)

            if agent.cfg.voix_actif and agent.voix is not None and info_v.disponible:
                try:
                    agent.dire(message, bloquant=False)
                except Exception as e:
                    print(f"[Voix] Salutation impossible : {e}")

            self.maj_emotions()
            self._pastille(agent.model_name, GOOD)
            self.charger_modeles()
            self.rafraichir()
            self._boucle_ui()

        self.run_async(lambda: AgentCerebralNyx(**kw), apres, "Chargement de Nyx…")

    def quitter(self):
        if self.occupe and not messagebox.askyesno("Quitter",
                                                   "Une opération est en cours. Quitter quand même ?"):
            return
        if self.agent:
            try:
                self.agent.arreter_pulsions()
                self.agent.arreter_voix()
            except Exception:
                pass
        sys.stdout = sys.__stdout__
        self.console.fermer()
        self.sauver_reglages()
        self.destroy()


if __name__ == "__main__":
    NyxApp().mainloop()

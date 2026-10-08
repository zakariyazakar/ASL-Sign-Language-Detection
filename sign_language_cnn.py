"""
Sign Language Detection - Webcam Amélioré
==========================================
Version complète avec :
  - Stabilisation des prédictions (vote sur 5 frames)
  - Construction de mots lettre par lettre
  - Barre de confiance visuelle
  - Affichage Top-3 probabilités
  - Prédiction toutes les 3 frames (fluidité)
  - Fond de la ROI grisé pour meilleure détection

Usage:
    python sign_language_webcam.py

Touches:
    Q     → Quitter
    ESPACE → Ajouter un espace au mot
    BACK  → Effacer la dernière lettre (touche B)
    C     → Effacer le mot entier
    S     → Sauvegarder une capture écran
"""

import os
import json
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = 64
MODEL_PATH  = "sign_model.h5"
LABELS_PATH = "class_labels.json"

BOX_SIZE         = 300       # taille de la zone ROI
PRED_EVERY_N     = 3         # prédire 1 frame sur N
STABILITY_FRAMES = 5         # nb de prédictions pour le vote
HOLD_FRAMES      = 20        # frames consécutives avant d'écrire une lettre

# Couleurs BGR
COLOR_GREEN  = (0, 220, 100)
COLOR_BLUE   = (255, 140, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_RED    = (0, 60, 220)
COLOR_GRAY   = (180, 180, 180)
COLOR_YELLOW = (0, 210, 255)


# ─────────────────────────────────────────────
#  CHARGEMENT MODÈLE
# ─────────────────────────────────────────────
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : '{MODEL_PATH}'. Entraînez d'abord le modèle.")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels introuvables : '{LABELS_PATH}'.")

    print("⏳ Chargement du modèle...")
    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    print(f"✅ Modèle chargé — {len(class_names)} classes : {class_names}\n")
    return model, class_names


# ─────────────────────────────────────────────
#  PRÉTRAITEMENT IMAGE
# ─────────────────────────────────────────────
def preprocess(roi_bgr):
    """Convertit la ROI en tenseur normalisé 1×64×64×3."""
    roi_rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_res  = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    roi_norm = roi_res.astype("float32") / 255.0
    return np.expand_dims(roi_norm, axis=0)


# ─────────────────────────────────────────────
#  DESSIN HUD
# ─────────────────────────────────────────────
def draw_roi_box(frame, x1, y1, x2, y2, color=COLOR_GREEN):
    """Dessine la boîte ROI avec coins stylisés."""
    thickness = 2
    corner    = 30
    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 1)
    # Coins
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), color, thickness+1)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), color, thickness+1)


def draw_confidence_bar(frame, x, y, w, confidence, color=COLOR_GREEN):
    """Barre de progression de la confiance."""
    bar_w = int(w * confidence / 100)
    cv2.rectangle(frame, (x, y), (x + w, y + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 18), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 18), COLOR_GRAY, 1)
    cv2.putText(frame, f"{confidence:.1f}%", (x + w + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


def draw_top3(frame, probs, class_names, x, y):
    """Affiche le top-3 des probabilités."""
    top3 = np.argsort(probs)[::-1][:3]
    cv2.putText(frame, "Top-3 :", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1)
    for i, idx in enumerate(top3):
        lbl  = class_names[idx]
        prob = probs[idx] * 100
        col  = COLOR_GREEN if i == 0 else COLOR_GRAY
        bar_w = int(120 * probs[idx])
        cv2.rectangle(frame, (x, y + 10 + i*28), (x + bar_w, y + 26 + i*28), col, -1)
        cv2.putText(frame, f"{lbl:<8} {prob:5.1f}%",
                    (x + 130, y + 24 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


def draw_word_panel(frame, word, h, w):
    """Panneau en bas affichant le mot construit."""
    panel_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Mot :", (15, h - panel_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GRAY, 1)
    display = word if word else "_"
    cv2.putText(frame, display, (80, h - panel_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_YELLOW, 2)


def draw_hold_progress(frame, x1, y2, hold_count, hold_max, color=COLOR_GREEN):
    """Barre de progression 'hold' avant écriture de la lettre."""
    w   = BOX_SIZE
    pct = min(hold_count / hold_max, 1.0)
    bar = int(w * pct)
    cv2.rectangle(frame, (x1, y2 + 8), (x1 + w, y2 + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y2 + 8), (x1 + bar, y2 + 18), color, -1)


def draw_controls(frame, w):
    """Légende des touches en haut à droite."""
    controls = ["Q=Quitter", "B=Suppr", "C=Effacer", "S=Capture"]
    for i, txt in enumerate(controls):
        cv2.putText(frame, txt, (w - 130, 22 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)


# ─────────────────────────────────────────────
#  WEBCAM PRINCIPALE
# ─────────────────────────────────────────────
def predict_webcam():
    model, class_names = load_model_and_labels()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam introuvable.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("📷 Webcam ouverte.\n")
    print("  HOLD la main immobile → la lettre s'ajoute automatiquement")
    print("  B → supprimer dernière lettre")
    print("  C → effacer le mot")
    print("  S → sauvegarder capture")
    print("  Q → quitter\n")

    # État
    predictions_buffer = deque(maxlen=STABILITY_FRAMES)
    stable_label   = ""
    last_stable    = ""
    hold_count     = 0
    word           = ""
    frame_count    = 0
    probs_display  = np.zeros(len(class_names))
    fps_time       = time.time()
    fps            = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # miroir naturel
        h, w  = frame.shape[:2]
        frame_count += 1

        # ── Zone ROI ──────────────────────────────
        x1 = (w - BOX_SIZE) // 2
        y1 = (h - BOX_SIZE) // 2
        x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
        roi = frame[y1:y2, x1:x2]

        # ── Prédiction toutes les N frames ────────
        if frame_count % PRED_EVERY_N == 0:
            inp   = preprocess(roi)
            probs = model.predict(inp, verbose=0)[0]
            probs_display = probs
            pred_idx = np.argmax(probs)
            predictions_buffer.append(class_names[pred_idx])

        # ── Vote de stabilité ─────────────────────
        if len(predictions_buffer) == STABILITY_FRAMES:
            stable_label = max(set(predictions_buffer),
                               key=predictions_buffer.count)

        confidence = float(probs_display[class_names.index(stable_label)]) * 100 \
                     if stable_label and stable_label in class_names else 0.0

        # ── Logique HOLD pour écriture ────────────
        special = stable_label in ("nothing", "")
        if stable_label == last_stable and not special and confidence > 70:
            hold_count += 1
        else:
            hold_count = 0
            last_stable = stable_label

        if hold_count >= HOLD_FRAMES:
            hold_count = 0
            if stable_label == "space":
                word += " "
            elif stable_label == "del":
                word = word[:-1]
            elif stable_label not in ("nothing", ""):
                word += stable_label
            last_stable = ""   # reset pour attendre un nouveau signe

        # ── FPS ───────────────────────────────────
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = now

        # ─────────────────────────────────────────
        #  DESSIN HUD
        # ─────────────────────────────────────────

        # Fond semi-transparent panneau gauche
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Boîte ROI
        box_color = COLOR_GREEN if confidence > 70 else COLOR_RED
        draw_roi_box(frame, x1, y1, x2, y2, box_color)

        # Label principal au-dessus de la boîte
        if stable_label and stable_label not in ("nothing", ""):
            label_txt = stable_label.upper()
            (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
            cv2.putText(frame, label_txt,
                        (x1 + (BOX_SIZE - lw) // 2, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, box_color, 4)

        # Barre de confiance sous la boîte
        draw_confidence_bar(frame, x1, y2 + 25, BOX_SIZE - 60, confidence, box_color)

        # Barre de hold
        if hold_count > 0:
            draw_hold_progress(frame, x1, y2 + 45, hold_count, HOLD_FRAMES, COLOR_YELLOW)
            cv2.putText(frame, "Hold...", (x1 + BOX_SIZE // 2 - 30, y2 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

        # Top-3 (panneau gauche)
        draw_top3(frame, probs_display, class_names, 10, 30)

        # FPS
        cv2.putText(frame, f"FPS: {fps}", (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1)

        # Contrôles (haut droite)
        draw_controls(frame, w)

        # Mot construit (panneau bas)
        draw_word_panel(frame, word, h, w)

        # ─────────────────────────────────────────
        #  AFFICHAGE
        # ─────────────────────────────────────────
        cv2.imshow("Sign Language Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b") or key == 8:   # B ou Backspace
            word = word[:-1]
        elif key == ord("c"):
            word = ""
        elif key == ord("s"):
            fname = f"capture_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(fname, frame)
            print(f"📸 Capture sauvegardée : {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session terminée. Mot final : '{word}'")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    predict_webcam()
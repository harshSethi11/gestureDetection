import sys
import time
import math
from collections import deque, Counter

import cv2
import numpy as np

# ---------- TTS (non-blocking on main thread via startLoop/iterate) ----------
_TTS = None
_TTS_OK = False

def tts_init():
    """Initialize pyttsx3 SAPI5; fall back to beep if missing."""
    global _TTS, _TTS_OK
    try:
        import pyttsx3  # requires comtypes & pywin32 on Windows
        _TTS = pyttsx3.init(driverName='sapi5')
        _TTS.setProperty('rate', 175)
        _TTS.setProperty('volume', 1.0)
        # Choose a voice if you want:
        # for v in _TTS.getProperty("voices"):
        #     if "en-IN" in v.name or "Zira" in v.name:
        #         _TTS.setProperty("voice", v.id); break
        _TTS.startLoop(False)   # non-blocking
        _TTS_OK = True
    except Exception:
        _TTS = None
        _TTS_OK = False

def tts_iterate():
    if _TTS_OK and _TTS:
        try:
            _TTS.iterate()
        except Exception:
            pass

def tts_say(msg: str):
    if _TTS_OK and _TTS:
        try:
            _TTS.say(msg)
            return True
        except Exception:
            return False
    return False

def tts_shutdown():
    if _TTS_OK and _TTS:
        try:
            _TTS.endLoop()
        except Exception:
            pass

def speak_or_beep(message="Command"):
    """Prefer voice; otherwise beep once."""
    ok = tts_say(message)
    if not ok:
        try:
            import winsound
            winsound.Beep(950, 500)
        except Exception:
            pass

# ---------- Try MediaPipe ----------
try:
    import mediapipe as mp
except Exception:
    print("[ERROR] mediapipe is not installed. Install it with:")
    print("        pip install mediapipe")
    sys.exit(1)

# ---------------- Config ----------------
CAM_INDEX = 0
FRAME_WIDTH = 960              # set None to keep native
SMOOTH_WINDOW = 7              # majority vote window over last N predictions
STABLE_FRAMES = 5              # min stable frames to accept a new gesture
ANNOUNCE_COOLDOWN = 1.5        # seconds between voice/beep announcements

# HUD colors
GREEN = (60, 220, 60)
YELLOW = (40, 180, 255)
RED = (36, 36, 255)
GRAY = (180, 180, 180)
WHITE = (255, 255, 255)

# Mapping pretty labels to phrases (for TTS)
SPOKEN = {
    "STOP": "Stop",
    "FIST": "Start",
    "POINT": "Pick",
    "THUMBS_UP": "Drop"
}

# ------- Robust finger geometry (drop-in replacement) -------
TIP_IDS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky tips
PIP_IDS = [3, 7, 11, 15, 19]     # PIP joints (thumb uses IP)
MCP_IDS = [2, 5, 9, 13, 17]      # MCP joints (thumb MCP = 2)
WRIST_ID = 0

MARGIN_Y = 0.02           # non-thumb slack
MIN_EXT_LEN = 0.065       # non-thumb min length to call "extended"
THUMB_MIN_LEN = 0.095     # thumb must be pretty long to consider at all
THUMB_UP_MAX_ANGLE = 20   # tighter: within 20° of vertical
THUMB_UP_MAX_HORIZ = 0.030  # horizontal drift tolerance for vertical thumb
PALM_OPEN_RATIO = 0.85


def _len(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def _finger_extended(landmarks, tip_id, pip_id, mcp_id):
    """Non-thumb: tip above pip (y smaller) and tip->mcp long enough."""
    tip, pip, mcp = landmarks[tip_id], landmarks[pip_id], landmarks[mcp_id]
    up_by_y = (tip.y < (pip.y - MARGIN_Y)) and (pip.y < (mcp.y - MARGIN_Y))
    long_enough = _len(tip.x, tip.y, mcp.x, mcp.y) >= MIN_EXT_LEN
    return bool(up_by_y and long_enough)

def _thumb_extended(landmarks):
    """
    Count thumb 'extended' only for clear sideways extension (open palm),
    NOT for vertical (that's handled by _thumb_pointing_up).
    """
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    dx = abs(tip.x - mcp.x)
    dy = tip.y - mcp.y
    length = math.hypot(dx, dy)
    # Strong sideways spread only
    return (dx > 0.070) and (length > 0.090)


def _thumb_pointing_up(landmarks):
    """
    True only if thumb is long, nearly vertical, minimal sideways drift,
    and the tip is clearly above MCP.
    """
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    vx, vy = (tip.x - mcp.x), (tip.y - mcp.y)
    length = math.hypot(vx, vy)
    if length < THUMB_MIN_LEN:
        return False

    # angle to vertical up
    up = np.array([0.0, -1.0], dtype=np.float32)
    v = np.array([vx, vy], dtype=np.float32)
    ang = float(np.degrees(np.arccos(np.clip((v @ up) / (np.linalg.norm(v) + 1e-6), -1.0, 1.0))))
    if ang > THUMB_UP_MAX_ANGLE:
        return False

    # minimal horizontal deviation & tip clearly above MCP
    if abs(vx) > THUMB_UP_MAX_HORIZ:
        return False
    if tip.y > (mcp.y - 0.040):  # need some vertical margin
        return False

    return True


def fingers_up(landmarks):
    """Return dict of which fingers are extended."""
    return {
        'thumb':  _thumb_extended(landmarks),
        'index':  _finger_extended(landmarks, TIP_IDS[1], PIP_IDS[1], MCP_IDS[1]),
        'middle': _finger_extended(landmarks, TIP_IDS[2], PIP_IDS[2], MCP_IDS[2]),
        'ring':   _finger_extended(landmarks, TIP_IDS[3], PIP_IDS[3], MCP_IDS[3]),
        'pinky':  _finger_extended(landmarks, TIP_IDS[4], PIP_IDS[4], MCP_IDS[4]),
    }

def detect_gesture(landmarks, _handedness_label_unused):
    """
    Returns: STOP, FIST, POINT, THUMBS_UP, or None.
    """
    up = fingers_up(landmarks)
    idx, mid, rng, pky = up['index'], up['middle'], up['ring'], up['pinky']

    # --- FIST: all non-thumb down, and thumb NOT clearly extended ---
    # Stricter thumb 'not extended' (close to palm)
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    thumb_dx = abs(tip.x - mcp.x)
    thumb_dy = tip.y - mcp.y
    thumb_len = math.hypot(thumb_dx, thumb_dy)
    thumb_not_extended = (thumb_dx < 0.045) and (thumb_len < 0.075 or thumb_dy >= -0.02)

    # Compactness fallback: fingertips cluster near wrist when fist is made
    wrist = landmarks[WRIST_ID]
    tip_ids_non_thumb = TIP_IDS[1:]  # index..pinky
    tip_dists = [math.hypot(landmarks[t].x - wrist.x, landmarks[t].y - wrist.y) for t in tip_ids_non_thumb]
    avg_tip_to_wrist = sum(tip_dists) / len(tip_dists)

    all_four_down = not idx and not mid and not rng and not pky
    if all_four_down and (thumb_not_extended or avg_tip_to_wrist < 0.13):
        return "FIST"

    # --- STOP: open palm (ignore thumb) ---
    if idx and mid and rng and pky:
        idx_len = _len(landmarks[TIP_IDS[1]].x, landmarks[TIP_IDS[1]].y,
                       landmarks[MCP_IDS[1]].x, landmarks[MCP_IDS[1]].y)
        mid_len = _len(landmarks[TIP_IDS[2]].x, landmarks[TIP_IDS[2]].y,
                       landmarks[MCP_IDS[2]].x, landmarks[MCP_IDS[2]].y)
        if (idx_len + mid_len) / (2 * MIN_EXT_LEN) >= PALM_OPEN_RATIO:
            return "STOP"

    # --- POINT: index up, others down (thumb ignored) ---
    if idx and not (mid or rng or pky):
        return "POINT"

    # --- THUMBS_UP: thumb vertical up, others down ---
    if _thumb_pointing_up(landmarks) and not (idx or mid or rng or pky):
         return "THUMBS_UP"

# ------------- Main -------------
def main():
    # Init TTS once
    tts_init()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        tts_shutdown()
        return

    # Resize hint
    if FRAME_WIDTH is not None:
        ret, fr = cap.read()
        if ret:
            scale = FRAME_WIDTH / fr.shape[1]
            target_h = int(fr.shape[0] * scale)
        else:
            target_h = None
    else:
        target_h = None

    # Smoothing, stability, and announcement
    recent = deque(maxlen=SMOOTH_WINDOW)
    stable_label = None
    stable_count = 0
    last_announced = None
    last_announce_time = 0.0

    fps_t0 = time.time()
    fps_counter = 0
    fps_val = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            tts_iterate()  # keep pyttsx3 event loop alive
            ok, frame = cap.read()
            if not ok:
                continue

            # Optional mirror for user-facing view
            frame = cv2.flip(frame, 1)

            if FRAME_WIDTH is not None:
                frame = cv2.resize(frame, (FRAME_WIDTH, target_h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label = None
            handed = None
            if res.multi_hand_landmarks and res.multi_handedness:
                hand_landmarks = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label  # "Left" or "Right"
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )
                # Classify gesture
                label = detect_gesture(hand_landmarks.landmark, handed)

            # Smoothing + stability
            recent.append(label if label else "NONE")
            common = Counter(recent).most_common(1)[0][0]  # mode
            if common != stable_label:
                stable_label = common
                stable_count = 1
            else:
                stable_count += 1

            # Announce only when stable enough and cooldown elapsed
            to_display = "—"
            if stable_label != "NONE" and stable_count >= STABLE_FRAMES:
                to_display = stable_label
                now = time.time()
                if (stable_label != last_announced) or (now - last_announce_time >= ANNOUNCE_COOLDOWN):
                    # Speak/beep
                    speak_or_beep(SPOKEN.get(stable_label, stable_label.title()))
                    last_announced = stable_label
                    last_announce_time = now

            # FPS
            fps_counter += 1
            if fps_counter >= 10:
                now = time.time()
                dt = now - fps_t0
                if dt > 0:
                    fps_val = fps_counter / dt
                fps_counter = 0
                fps_t0 = now

            # HUD
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesture: {to_display}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        GREEN if to_display not in ("—", "NONE") else GRAY, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps_val:.1f}   [Q] Quit",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

            cv2.imshow("Hand Signal Control (Stop/Fist/Point/Thumbs-Up)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()
    tts_shutdown()

if __name__ == "__main__":
    main()

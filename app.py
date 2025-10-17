import math
from collections import deque, Counter
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# Streamlit UI Setup

st.set_page_config(page_title="Hand Gesture Detection", layout="centered")
st.title("🖐️ Real-time Hand Gesture Detection")
st.markdown(
    "Detects **STOP**, **FIST**, **POINT**, and **THUMBS_UP** using MediaPipe. "
    "Shows a stable label after smoothing over a short window."
)

# Config (tweak in sidebar) 
FRAME_WIDTH       = st.sidebar.slider("Frame width", 480, 1280, 720, 10)
SMOOTH_WINDOW     = st.sidebar.slider("Smoothing window (frames)", 3, 15, 7, 1)
STABLE_FRAMES     = st.sidebar.slider("Frames needed to confirm gesture", 3, 20, 5, 1)
SHOW_LANDMARKS    = st.sidebar.checkbox("Show hand landmarks", True)

st.sidebar.info("Tip: If camera doesn’t start, refresh or try another browser.")

# MediaPipe setup 
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Robust finger geometry (adapted from your code) 
TIP_IDS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky tips
PIP_IDS = [3, 7, 11, 15, 19]     # PIP joints (thumb uses IP)
MCP_IDS = [2, 5, 9, 13, 17]      # MCP joints (thumb MCP = 2)
WRIST_ID = 0

MARGIN_Y = 0.02           # non-thumb slack
MIN_EXT_LEN = 0.065       # non-thumb min length to call "extended"
THUMB_MIN_LEN = 0.095     # thumb must be pretty long to consider at all
THUMB_UP_MAX_ANGLE = 20   # within 20 degree of vertical
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
    """Sideways thumb extension (for open palm), not vertical thumbs-up."""
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    dx = abs(tip.x - mcp.x)
    dy = tip.y - mcp.y
    length = math.hypot(dx, dy)
    return (dx > 0.070) and (length > 0.090)

def _thumb_pointing_up(landmarks):
    """Thumb long, near-vertical, minimal sideways drift, tip above MCP."""
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    vx, vy = (tip.x - mcp.x), (tip.y - mcp.y)
    length = math.hypot(vx, vy)
    if length < THUMB_MIN_LEN:
        return False
    up = np.array([0.0, -1.0], dtype=np.float32)  # vertical up
    v = np.array([vx, vy], dtype=np.float32)
    ang = float(np.degrees(np.arccos(np.clip((v @ up) / (np.linalg.norm(v) + 1e-6), -1.0, 1.0))))
    if ang > THUMB_UP_MAX_ANGLE:
        return False
    if abs(vx) > THUMB_UP_MAX_HORIZ:
        return False
    if tip.y > (mcp.y - 0.040):
        return False
    return True

def fingers_up(landmarks):
    return {
        'thumb':  _thumb_extended(landmarks),
        'index':  _finger_extended(landmarks, TIP_IDS[1], PIP_IDS[1], MCP_IDS[1]),
        'middle': _finger_extended(landmarks, TIP_IDS[2], PIP_IDS[2], MCP_IDS[2]),
        'ring':   _finger_extended(landmarks, TIP_IDS[3], PIP_IDS[3], MCP_IDS[3]),
        'pinky':  _finger_extended(landmarks, TIP_IDS[4], PIP_IDS[4], MCP_IDS[4]),
    }

def detect_gesture(landmarks, _hand_label_unused):
    """
    Returns: 'STOP', 'FIST', 'POINT', 'THUMBS_UP', or None.
    """
    up = fingers_up(landmarks)
    idx, mid, rng, pky = up['index'], up['middle'], up['ring'], up['pinky']

    # FIST: all non-thumb down, and thumb NOT clearly extended 
    tip, mcp = landmarks[TIP_IDS[0]], landmarks[MCP_IDS[0]]
    thumb_dx = abs(tip.x - mcp.x)
    thumb_dy = tip.y - mcp.y
    thumb_len = math.hypot(thumb_dx, thumb_dy)
    thumb_not_extended = (thumb_dx < 0.045) and (thumb_len < 0.075 or thumb_dy >= -0.02)

    wrist = landmarks[WRIST_ID]
    tip_ids_non_thumb = TIP_IDS[1:]
    tip_dists = [math.hypot(landmarks[t].x - wrist.x, landmarks[t].y - wrist.y) for t in tip_ids_non_thumb]
    avg_tip_to_wrist = sum(tip_dists) / len(tip_dists)

    all_four_down = not idx and not mid and not rng and not pky
    if all_four_down and (thumb_not_extended or avg_tip_to_wrist < 0.13):
        return "FIST"

    #  STOP: open palm (ignore thumb)
    if idx and mid and rng and pky:
        idx_len = _len(landmarks[TIP_IDS[1]].x, landmarks[TIP_IDS[1]].y,
                       landmarks[MCP_IDS[1]].x, landmarks[MCP_IDS[1]].y)
        mid_len = _len(landmarks[TIP_IDS[2]].x, landmarks[TIP_IDS[2]].y,
                       landmarks[MCP_IDS[2]].x, landmarks[MCP_IDS[2]].y)
        if (idx_len + mid_len) / (2 * MIN_EXT_LEN) >= PALM_OPEN_RATIO:
            return "STOP"

    # POINT: index up, others down (thumb ignored)
    if idx and not (mid or rng or pky):
        return "POINT"

    # THUMBS_UP: thumb vertical up, others down
    if _thumb_pointing_up(landmarks) and not (idx or mid or rng or pky):
        return "THUMBS_UP"

    return None

# Video Transformer

class GestureTransformer(VideoTransformerBase):
    def __init__(self, frame_width: int, smooth_window: int, stable_frames: int, show_landmarks: bool):
        self.frame_width = frame_width
        self.smooth_window = smooth_window
        self.stable_frames = stable_frames
        self.show_landmarks = show_landmarks

        self.recent = deque(maxlen=self.smooth_window)
        self.stable_label = None
        self.stable_count = 0

        # FPS calc
        self.fps_t0 = time.time()
        self.fps_counter = 0
        self.fps_val = 0.0

        # Mediapipe Hands instance
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def __del__(self):
        if hasattr(self, "hands") and self.hands:
            self.hands.close()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Resize to desired width (keeping aspect)
        h, w = img.shape[:2]
        if self.frame_width and w != self.frame_width:
            scale = self.frame_width / w
            img = cv2.resize(img, (self.frame_width, int(h * scale)))
            h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        label = None
        if res.multi_hand_landmarks and res.multi_handedness:
            hand_landmarks = res.multi_hand_landmarks[0]
            handed_label = res.multi_handedness[0].classification[0].label  # "Left" or "Right"

            if self.show_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

            label = detect_gesture(hand_landmarks.landmark, handed_label)

        # Smoothing + stability
        self.recent.append(label if label else "NONE")
        common = Counter(self.recent).most_common(1)[0][0]  # mode
        if common != self.stable_label:
            self.stable_label = common
            self.stable_count = 1
        else:
            self.stable_count += 1

        # what to show
        to_display = "—"
        if self.stable_label != "NONE" and self.stable_count >= self.stable_frames:
            to_display = self.stable_label

        # FPS
        self.fps_counter += 1
        if self.fps_counter >= 10:
            now = time.time()
            dt = now - self.fps_t0
            if dt > 0:
                self.fps_val = self.fps_counter / dt
            self.fps_counter = 0
            self.fps_t0 = now

        # HUD
        cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
        color = (0, 255, 0) if to_display not in ("—", "NONE") else (200, 200, 200)
        cv2.putText(img, f"Gesture: {to_display}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {self.fps_val:.1f}", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Launch WebRTC Stream
# Some versions don’t expose WebRtcMode.LIVE; default mode works fine.

webrtc_streamer(
    key="gesture-demo",
    video_transformer_factory=lambda: GestureTransformer(
        frame_width=FRAME_WIDTH,
        smooth_window=SMOOTH_WINDOW,
        stable_frames=STABLE_FRAMES,
        show_landmarks=SHOW_LANDMARKS
    ),
    media_stream_constraints={"video": True, "audio": False},
)

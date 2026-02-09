# gaze_utils.py
# Improved, sensitive gaze utilities for MediaPipe FaceMesh (refined landmarks enabled)
# Requires: numpy, collections.deque, cv2

import numpy as np
import cv2
from collections import deque
from math import atan2, degrees

# NOTE on indices:
# This file assumes MediaPipe FaceMesh with refine_landmarks=True (iris landmarks available).
# Indices used here worked reliably in many setups; if they mismatch your MP version, adjust.
LEFT_IRIS = [474, 475, 476, 477]   # left eye iris points
RIGHT_IRIS = [469, 470, 471, 472]  # right eye iris points
LEFT_EYE_LR = [33, 133]            # left eye left & right corner (outer, inner)
RIGHT_EYE_LR = [362, 263]          # right eye left & right corner (inner, outer)
LEFT_EYE_TB = [159, 145]           # left eye top & bottom (for EAR)
RIGHT_EYE_TB = [386, 374]          # right eye top & bottom (for EAR)

# 3D model points for head pose (generic approximate)
_P3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),      # nose tip
    (0.0, -63.6, -12.5),  # chin
    (-43.3, 32.7, -26.0), # left eye left corner
    (43.3, 32.7, -26.0),  # right eye right corner
    (-28.9, -28.9, -24.1),# left mouth corner
    (28.9, -28.9, -24.1)  # right mouth corner
], dtype=np.float64)

# FaceMesh landmark indices used for head-pose 2D projection:
_HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,   # approximate — will map below
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

class GazeEstimator:
    def __init__(self,
                 history_len=9,
                 ema_alpha=0.35,
                 blink_ear_thresh=0.18,
                 pupil_detect_method="landmark",  # keep "landmark"
                 compensate_head_pose=True):
        self.history = deque(maxlen=history_len)
        self.ema_alpha = ema_alpha
        self.ema_value = None
        self.prev_direction = "CENTER"
        self.blink_thresh = blink_ear_thresh
        self.compensate_head_pose = compensate_head_pose

        # calibration placeholders (will be set by calibrate_session)
        self.left_center = 0.30   # expected mean when looking left
        self.center_center = 0.50
        self.right_center = 0.70
        self.lower_thresh = 0.43
        self.upper_thresh = 0.57

    # ---------- low-level helpers ----------
    def _landmark_xy(self, lm, width, height):
        return np.array([int(lm.x * width), int(lm.y * height)], dtype=int)

    def get_iris_center(self, landmarks, width, height, indices):
        pts = np.array([self._landmark_xy(landmarks[i], width, height) for i in indices])
        # use median for robustness
        center = np.median(pts, axis=0).astype(int)
        return center

    def get_eye_lr(self, landmarks, width, height, indices_lr):
        # return ordered (left_x, right_x)
        x1 = landmarks[indices_lr[0]].x * width
        x2 = landmarks[indices_lr[1]].x * width
        left = int(min(x1, x2))
        right = int(max(x1, x2))
        return left, right

    def eye_aspect_ratio(self, landmarks, width, height, tb_indices):
        # Simple vertical/horizontal measure using top and bottom vs corners
        try:
            top = self._landmark_xy(landmarks[tb_indices[0]], width, height)
            bottom = self._landmark_xy(landmarks[tb_indices[1]], width, height)
            # horizontal width approximate using eye lr
            # find nearby corners:
            # for left eye corners are 33 and 133, for right 362 and 263 (handled by caller)
            return abs(top[1] - bottom[1])  # pixel vertical distance (we normalize later)
        except Exception:
            return None

    # ---------- head pose (optional) ----------
    def estimate_head_pose(self, landmarks, width, height):
        # Build 2D image points from landmarks
        try:
            img_pts = np.array([
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["nose_tip"]], width, height),
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["chin"]], width, height),
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["left_eye_corner"]], width, height),
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["right_eye_corner"]], width, height),
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["left_mouth"]], width, height),
                self._landmark_xy(landmarks[_HEAD_POSE_LANDMARKS["right_mouth"]], width, height),
            ], dtype=np.float64)

            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

            ok, rvec, tvec = cv2.solvePnP(_P3D_MODEL_POINTS, img_pts, camera_matrix, dist_coeffs,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            rmat, _ = cv2.Rodrigues(rvec)
            # compute yaw from rotation matrix
            sy = -rmat[2, 0]
            yaw = degrees(atan2(rmat[1, 0], rmat[0, 0]))  # approximate yaw
            pitch = degrees(atan2(rmat[2, 1], rmat[2, 2]))
            roll = degrees(atan2(rmat[1, 2], rmat[1, 1]))
            return {"yaw": yaw, "pitch": pitch, "roll": roll}
        except Exception:
            return None

    # ---------- main gaze ratio computation ----------
    def compute_gaze_ratio(self, landmarks, width, height):
        """
        returns gaze_ratio in [0,1] where 0 = extreme left (eyes pointing left), 1 = extreme right.
        """
        try:
            # get iris centers
            l_iris = self.get_iris_center(landmarks, width, height, LEFT_IRIS)
            r_iris = self.get_iris_center(landmarks, width, height, RIGHT_IRIS)

            # get eye left/right bounds
            l_left, l_right = self.get_eye_lr(landmarks, width, height, LEFT_EYE_LR)
            r_left, r_right = self.get_eye_lr(landmarks, width, height, RIGHT_EYE_LR)

            # normalized ratios relative to each eye box
            l_ratio = (l_iris[0] - l_left) / max((l_right - l_left), 1)
            r_ratio = (r_iris[0] - r_left) / max((r_right - r_left), 1)

            combined = (l_ratio + r_ratio) / 2.0
            combined = float(np.clip(combined, 0.0, 1.0))

            # optional: head-pose compensation (simple linear correction from yaw)
            if self.compensate_head_pose:
                hp = self.estimate_head_pose(landmarks, width, height)
                if hp is not None:
                    # yaw positive may indicate face turned to right -> subtract yaw*factor
                    yaw = hp.get("yaw", 0.0)
                    # empirical factor: normalize yaw degrees (~ -30..30) to [-0.2..0.2]
                    correction = np.clip(-yaw / 180.0, -0.25, 0.25)
                    combined = np.clip(combined + correction, 0.0, 1.0)

            return combined
        except Exception:
            return None

    # ---------- blink detection ----------
    def is_blinking(self, landmarks, width, height):
        # Compute a relative vertical opening for both eyes and compare to threshold
        ly = self.eye_aspect_ratio(landmarks, width, height, LEFT_EYE_TB)
        ry = self.eye_aspect_ratio(landmarks, width, height, RIGHT_EYE_TB)

        # fallback if None
        if ly is None or ry is None:
            return False

        # Normalize by eye width
        l_left, l_right = self.get_eye_lr(landmarks, width, height, LEFT_EYE_LR)
        r_left, r_right = self.get_eye_lr(landmarks, width, height, RIGHT_EYE_LR)
        l_width = max((l_right - l_left), 1)
        r_width = max((r_right - r_left), 1)
        l_norm = ly / l_width
        r_norm = ry / r_width

        avg = (l_norm + r_norm) / 2.0
        return avg < self.blink_thresh

    # ---------- smoothing + hysteresis ----------
    def smooth_ratio(self, ratio):
        if ratio is None:
            return None
        # EMA
        if self.ema_value is None:
            self.ema_value = ratio
        else:
            self.ema_value = (self.ema_alpha * ratio) + (1.0 - self.ema_alpha) * self.ema_value
        # median buffer for short-term stability
        self.history.append(self.ema_value)
        return float(np.median(self.history))

    def calibrate_session(self, left_samples, center_samples, right_samples):
        """
        left_samples/center_samples/right_samples: lists of gaze_ratio floats collected while user looked left/center/right.
        This computes session-specific thresholds automatically.
        """
        left_mean = float(np.mean(left_samples)) if len(left_samples) else 0.25
        center_mean = float(np.mean(center_samples)) if len(center_samples) else 0.5
        right_mean = float(np.mean(right_samples)) if len(right_samples) else 0.65

        # set centers
        self.left_center = left_mean
        self.center_center = center_mean
        self.right_center = right_mean

        # thresholds are midpoints between center and left/right with small margin
        self.lower_thresh = float((left_mean + center_mean) / 2.0)
        self.upper_thresh = float((right_mean + center_mean) / 2.0)

        # small safety margins
        spread = max((self.upper_thresh - self.lower_thresh) * 0.1, 0.02)
        self.lower_thresh = max(0.0, self.lower_thresh - spread)
        self.upper_thresh = min(1.0, self.upper_thresh + spread)

        return {"left_mean": left_mean, "center_mean": center_mean, "right_mean": right_mean,
                "lower_thresh": self.lower_thresh, "upper_thresh": self.upper_thresh}

    def get_direction(self, smoothed_ratio):
        """
        Returns 'LEFT'|'CENTER'|'RIGHT' using hysteresis (relies on prev_direction).
        """
        if smoothed_ratio is None:
            return self.prev_direction  # keep previous if invalid

        # apply thresholds with hysteresis: if previously LEFT, require stronger movement to go CENTER/RIGHT
        if self.prev_direction == "LEFT":
            left_bound = self.lower_thresh + 0.02
            right_bound = self.upper_thresh
        elif self.prev_direction == "RIGHT":
            left_bound = self.lower_thresh
            right_bound = self.upper_thresh - 0.015
        else:
            left_bound = self.lower_thresh
            right_bound = self.upper_thresh

        if smoothed_ratio < left_bound:
            self.prev_direction = "LEFT"
        elif smoothed_ratio > right_bound:
            self.prev_direction = "RIGHT"
        else:
            self.prev_direction = "CENTER"

        return self.prev_direction

    # ---------- high level wrapper ----------
    def estimate(self, landmarks, width, height, skip_on_blink=True):
        """
        Single-call API: returns a dict:
        { "gaze_ratio": float, "smoothed": float, "direction": str, "blink": bool, "head_pose": dict or None }
        """
        blink = self.is_blinking(landmarks, width, height) if skip_on_blink else False
        if blink:
            return {"gaze_ratio": None, "smoothed": None, "direction": self.prev_direction, "blink": True, "head_pose": None}

        raw = self.compute_gaze_ratio(landmarks, width, height)
        if raw is None:
            return {"gaze_ratio": None, "smoothed": None, "direction": self.prev_direction, "blink": False, "head_pose": None}

        sm = self.smooth_ratio(raw)
        direction = self.get_direction(sm)
        head = self.estimate_head_pose(landmarks, width, height) if self.compensate_head_pose else None

        return {"gaze_ratio": raw, "smoothed": sm, "direction": direction, "blink": False, "head_pose": head}

    # ---------- debug drawing ----------
    def draw_debug(self, frame, landmarks, width, height, show_bbox=True):
        """Draw iris centers, eye boxes, head yaw text on frame (inplace)."""
        try:
            l_iris = self.get_iris_center(landmarks, width, height, LEFT_IRIS)
            r_iris = self.get_iris_center(landmarks, width, height, RIGHT_IRIS)
            l_l, l_r = self.get_eye_lr(landmarks, width, height, LEFT_EYE_LR)
            r_l, r_r = self.get_eye_lr(landmarks, width, height, RIGHT_EYE_LR)

            # dots
            cv2.circle(frame, tuple(l_iris), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(r_iris), 3, (0, 255, 0), -1)
            # eye boxes
            if show_bbox:
                cv2.line(frame, (l_l, 30), (l_r, 30), (255, 0, 0), 1)
                cv2.line(frame, (r_l, 50), (r_r, 50), (255, 0, 0), 1)

            # head pose text
            hp = self.estimate_head_pose(landmarks, width, height)
            if hp:
                cv2.putText(frame, f"Yaw:{hp['yaw']:.1f}", (10, height - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(frame, f"Pitch:{hp['pitch']:.1f}", (10, height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        except Exception:
            pass

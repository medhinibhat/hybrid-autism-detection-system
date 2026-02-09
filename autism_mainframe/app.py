import cv2
import time
import os
import pandas as pd
import mediapipe as mp
import tkinter as tk
from gaze_utils import GazeEstimator

# =========================
# CONFIG
# =========================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stimuli to run (name, path, expected_target)
# expected_target: "LEFT" means healthy user should look LEFT for this stimulus (bio)
#                  "RIGHT" means healthy user should look RIGHT for this stimulus (social)
STIMULI = [
    ("bio_vs_scrambled", "stimuli/bio_vs_scrambled.mp4", "LEFT"),
    ("shapes_vs_social", "stimuli/shapes_vs_social.mp4", "RIGHT"),
]

# Webcam index
WEBCAM_IDX = 0

# Get screen resolution
root = tk.Tk()
SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()
root.destroy()

# Initialize MediaPipe & GazeEstimator
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1,
                             min_detection_confidence=0.6, min_tracking_confidence=0.6)

gaze_estimator = GazeEstimator(history_len=9, ema_alpha=0.35,
                               blink_ear_thresh=0.16, compensate_head_pose=True)

# =========================
# Helpers
# =========================
def resize_to_screen(frame, screen_w, screen_h):
    """Resize a frame to fit within the screen, keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)  # keep aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_stimulus(name, video_path, expected_target):
    """
    Play stimulus and capture gaze using gaze_estimator.
    Returns the csv path and summary dict.
    """
    print(f"\n[INFO] Starting stimulus: {name} -> {video_path} (expect {expected_target})")
    cap_vid = cv2.VideoCapture(video_path)
    cap_cam = cv2.VideoCapture(WEBCAM_IDX)
    if not cap_vid.isOpened():
        raise IOError(f"Could not open stimulus video: {video_path}")
    if not cap_cam.isOpened():
        raise IOError(f"Could not open webcam index {WEBCAM_IDX}")

    frame_count = 0
    target_count = 0
    gaze_log = []

    start_ts = time.time()
    window_name_vid = f"Stimulus: {name}"
    window_name_cam = "Gaze Tracking"

    while cap_vid.isOpened() and cap_cam.isOpened():
        ret_vid, stim_frame = cap_vid.read()
        ret_cam, cam_frame = cap_cam.read()
        if not ret_vid:
            break  # stimulus finished
        if not ret_cam:
            print("[WARN] Webcam frame not available")
            break

        # flip camera so left/right matches user's left/right
        cam_frame = cv2.flip(cam_frame, 1)
        h, w, _ = cam_frame.shape

        # FaceMesh processing
        rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze_dir = None
        gaze_ratio = None
        smoothed = None
        blink = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            info = gaze_estimator.estimate(lm, w, h, skip_on_blink=True)
            gaze_ratio = info.get("gaze_ratio")
            smoothed = info.get("smoothed")
            gaze_dir = info.get("direction")
            blink = info.get("blink", False)

            # draw debug overlays
            gaze_estimator.draw_debug(cam_frame, lm, w, h)

            # annotate values
            cv2.putText(cam_frame, f"Gaze: {gaze_dir}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if smoothed is not None:
                cv2.putText(cam_frame, f"Ratio: {smoothed:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            if blink:
                cv2.putText(cam_frame, "Blink", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # counting & logging
        frame_count += 1
        is_target = False
        if gaze_dir is not None and not blink:
            is_target = (gaze_dir == expected_target)
            if is_target:
                target_count += 1

        gaze_log.append({
            "timestamp": time.time(),
            "gaze_ratio_raw": gaze_ratio,
            "gaze_ratio_smoothed": smoothed,
            "gaze_direction": gaze_dir,
            "blink": blink,
            "is_target": is_target
        })

        # live score overlay (percentage)
        pct = (target_count / frame_count) * 100 if frame_count > 0 else 0.0
        cv2.putText(stim_frame, f"Target%: {pct:5.1f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)

        # resize to fit screen
        stim_frame_resized = resize_to_screen(stim_frame, SCREEN_W, SCREEN_H)

        # show frames
        cv2.imshow(window_name_vid, stim_frame_resized)
        cv2.imshow(window_name_cam, cam_frame)

        # break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap_vid.release()
    cap_cam.release()
    cv2.destroyWindow(window_name_vid)
    cv2.destroyWindow(window_name_cam)

    # save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"gaze_log_{name}.csv")
    pd.DataFrame(gaze_log).to_csv(csv_path, index=False)
    print(f"[INFO] Saved gaze log -> {csv_path}")

    # compute final score for this stimulus (fraction of non-blink frames that were on-target)
    df = pd.DataFrame(gaze_log)
    usable = df[~df['blink']]
    if len(usable) == 0:
        score = 0.0
    else:
        score = usable['is_target'].sum() / len(usable)

    duration = time.time() - start_ts
    summary = {"name": name, "csv": csv_path, "frames": frame_count, "usable_frames": len(usable),
               "target_frames": int(usable['is_target'].sum()), "score": float(score), "duration_s": duration}
    print(f"[INFO] {name} -> score: {score:.3f} usable_frames: {len(usable)} total_frames: {frame_count}")
    return summary


def combine_scores(summaries, weights=None):
    """Combine per-stimulus scores into a single autism risk."""
    if weights is None:
        weights = {s['name']: 1.0 for s in summaries}
    # normalize weights
    total_w = sum(weights.values())
    combined = 0.0
    total = 0.0
    for s in summaries:
        w = weights.get(s['name'], 1.0)
        combined += s['score'] * w
        total += w
    avg_score = combined / total if total > 0 else 0.0
    # autism risk: lower gaze-to-social/biological -> higher risk
    autism_risk = 1.0 - avg_score
    return {"avg_score": avg_score, "autism_risk": autism_risk}


# =========================
# RUN ALL STIMULI
# =========================
if __name__ == "__main__":
    summaries = []
    for name, path, expected in STIMULI:
        if not os.path.exists(path):
            print(f"[WARN] Stimulus missing: {path}  — skipping {name}")
            continue
        summary = run_stimulus(name, path, expected)
        summaries.append(summary)

    # combine results
    if summaries:
        result = combine_scores(summaries)
        print("\n=== PER-STIMULUS SUMMARY ===")
        for s in summaries:
            print(f"{s['name']}: score={s['score']:.3f} usable={s['usable_frames']} dur={s['duration_s']:.1f}s csv={s['csv']}")
        print(f"\nFINAL AVERAGE SCORE (higher = more social/biological looking): {result['avg_score']:.3f}")
        print(f"FINAL AUTISM RISK (0 low — 1 high): {result['autism_risk']:.3f}")
    else:
        print("[WARN] No stimuli were run. Make sure stimulus files exist in the stimuli/ folder.")

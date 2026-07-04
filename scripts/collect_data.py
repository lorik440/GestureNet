from pathlib import Path
import cv2
import mediapipe as mp
import csv
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_legacy = PROJECT_ROOT / "data.csv"
_data_dir = PROJECT_ROOT / "data"
# If a legacy root data.csv exists and no new data/data.csv, use legacy file to avoid duplication.
if _legacy.exists() and not (_data_dir / "data.csv").exists():
    DATA_PATH = _legacy
else:
    _data_dir.mkdir(exist_ok=True)
    DATA_PATH = _data_dir / "data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

label = input("Enter gesture label: ").strip().lower()
mode  = input("One hand or two hands? (1/2): ").strip()
two_hand = mode == "2"
samples_to_collect = 400
count = 0

print(f"\nCollecting '{label}' ({'two-hand' if two_hand else 'one-hand'}) — press ESC to stop early\n")
time.sleep(2)


def extract_hand(lms):
    wrist = np.array(lms[0])
    scale = np.linalg.norm(wrist[:2] - np.array(lms[12])[:2]) + 1e-6
    return np.array([(c - wrist[i % 3]) / scale for pt in lms for i, c in enumerate(pt)], dtype=np.float32)


with open(DATA_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    while count < samples_to_collect:
        ret, frame = cap.read()
        if not ret:
            continue

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected = results.multi_hand_landmarks
        handedness = results.multi_handedness

        row = None

        if two_hand:
            if detected and len(detected) == 2:
                pairs = sorted(
                    zip(detected, handedness),
                    key=lambda x: 0 if x[1].classification[0].label == "Left" else 1
                )
                lms0 = [(lm.x, lm.y, lm.z) for lm in pairs[0][0].landmark]
                lms1 = [(lm.x, lm.y, lm.z) for lm in pairs[1][0].landmark]
                feat = np.concatenate([extract_hand(lms0), extract_hand(lms1)])
                row = feat.tolist() + [label]
                for hl, _ in pairs:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        else:
            if detected:
                lms = [(lm.x, lm.y, lm.z) for lm in detected[0].landmark]
                feat = np.concatenate([extract_hand(lms), np.zeros(63, dtype=np.float32)])
                row = feat.tolist() + [label]
                mp_draw.draw_landmarks(frame, detected[0], mp_hands.HAND_CONNECTIONS)

        if row is not None:
            writer.writerow(row)
            f.flush()
            count += 1
            time.sleep(0.12)

        cv2.putText(frame, f"{label}: {count}/{samples_to_collect}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done — {count} samples collected for '{label}'")

import os, re, sqlite3, cv2, torch
os.chdir("/home/subash/anpr")          # temporary: weights and videos still live here
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- settings ----------------
VIDEO      = "videos/dl.mp4"
YOLO_PATH  = "runs/detect/train/weights/best.pt"
TROCR_PATH = os.path.expanduser("~/anpr/trocr_plate")
CLEAR_ALL  = True
OCR_EVERY  = 5
MIN_WIDTH  = 60
DET_CONF   = 0.25
TRIM_EXTRA = True
MIN_CONF   = 0.80
MIN_VOTES  = 1
# ------------------------------------------

if "yolo" not in globals():
    yolo = YOLO(YOLO_PATH)
if "trocr" not in globals():
    proc  = TrOCRProcessor.from_pretrained(TROCR_PATH)
    trocr = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH).to(DEVICE).eval()

TO_DIGIT  = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2'}
TO_LETTER = {v: k for k, v in TO_DIGIT.items()}
PATTERNS  = [re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]?[A-Z]{1,3}\d{4}$')]   # Indian only
TEMPLATES = {
    8:  ["LLDLDDDD"],
    9:  ["LLDDLDDDD", "LLDLLDDDD"],
    10: ["LLDDLLDDDD", "LLDLLLDDDD"],
}

def is_valid(t):
    return any(p.match(t) for p in PATTERNS)

def _apply(tpl, t):
    return ''.join((TO_LETTER if k == "L" else TO_DIGIT).get(c, c) for k, c in zip(tpl, t))

def _snap(t):
    if is_valid(t):
        return t
    best = None
    for tpl in TEMPLATES.get(len(t), []):
        cand = _apply(tpl, t)
        if is_valid(cand):
            d = sum(a != b for a, b in zip(cand, t))
            if best is None or d < best[0]:
                best = (d, cand)
    return best[1] if best else None

def fix(t):
    s = _snap(t)
    if s:
        return s
    if TRIM_EXTRA and len(t) > 5:
        s = _snap(t[1:])
        if s:
            return s
    return t

def vote(reads):
    pool = [r for r in reads if is_valid(r[0])] or [r for r in reads if 8 <= len(r[0]) <= 11]
    if not pool:
        return None, 0, 0.0
    L = Counter(len(t) for t, _ in pool).most_common(1)[0][0]
    same = [(t, c) for t, c in pool if len(t) == L]
    text = ''.join(Counter(t[i] for t, _ in same).most_common(1)[0][0] for i in range(L))
    return text, len(same), sum(c for _, c in same) / len(same)

def read_plate(crop):
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pv = proc(pil, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        out = trocr.generate(pv, max_length=20, num_beams=8, num_return_sequences=5,
                             output_scores=True, return_dict_in_generate=True)
    cands = proc.batch_decode(out.sequences, skip_special_tokens=True)
    probs = torch.exp(out.sequences_scores).tolist()
    texts = [fix(re.sub(r'[^A-Z0-9]', '', c.upper())) for c in cands]
    for t, p in zip(texts, probs):
        if is_valid(t):
            return t, p
    return texts[0], probs[0]

db = sqlite3.connect("plates.db")
db.execute("""CREATE TABLE IF NOT EXISTS vehicle_plates(
  id INTEGER PRIMARY KEY, plate_text TEXT, valid INTEGER, votes INTEGER,
  avg_confidence REAL, timestamp TEXT, video_path TEXT, track_id INTEGER, crop_path TEXT)""")
if CLEAR_ALL:
    db.execute("DELETE FROM vehicle_plates")
else:
    db.execute("DELETE FROM vehicle_plates WHERE video_path = ?", (VIDEO,))
db.commit()

def process_video(src):
    global DEBUG
    assert os.path.exists(src), f"Video not found: {src}"
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"Input: {src} | {w}x{h} | {fps:.0f} fps | {total} frames | {os.path.getsize(src)/1e6:.1f} MB")

    stem = os.path.splitext(os.path.basename(src))[0]
    out = f"output_{stem}.mp4"
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    os.makedirs("crops", exist_ok=True)

    yolo.predictor = None
    reads, best_crop, frame_no = defaultdict(list), {}, 0

    for res in yolo.track(source=src, stream=True, persist=True,
                          tracker="bytetrack.yaml", conf=DET_CONF, verbose=False):
        frame_no += 1
        frame = res.orig_img.copy()
        if res.boxes.id is not None:
            for (x1, y1, x2, y2), tid in zip(res.boxes.xyxy.int().tolist(),
                                              res.boxes.id.int().tolist()):
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if tid not in best_crop or area > best_crop[tid][0]:
                    best_crop[tid] = (area, crop.copy())
                if frame_no % OCR_EVERY == 0 and (x2 - x1) >= MIN_WIDTH:
                    text, conf = read_plate(crop)
                    if 8 <= len(text) <= 11:
                        reads[tid].append((text, conf))
                label = vote(reads[tid])[0] if reads[tid] else None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{tid} {label or '...'}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(frame)
    writer.release()

    # extra vote: read each track's largest crop once more
    for tid, (_, c) in best_crop.items():
        if c.shape[1] >= MIN_WIDTH:
            t, cf = read_plate(c)
            if 8 <= len(t) <= 11:
                reads[tid].append((t, cf))

    DEBUG = (reads, best_crop)
    seen = set()
    for tid, rs in reads.items():
        text, n, conf = vote(rs)
        if not text or text in seen:
            continue
        seen.add(text)
        crop_path = f"crops/{stem}_track{tid}.jpg"
        cv2.imwrite(crop_path, best_crop[tid][1])
        db.execute("""INSERT INTO vehicle_plates(plate_text,valid,votes,avg_confidence,
                      timestamp,video_path,track_id,crop_path) VALUES(?,?,?,?,?,?,?,?)""",
                   (text, int(is_valid(text)), n, conf,
                    datetime.now().isoformat(), src, tid, crop_path))
    db.commit()
    print(f"{frame_no} frames processed, {len(seen)} vehicles stored -> {out}")

if __name__ == "__main__":
    process_video(VIDEO)
    print(pd.read_sql("""SELECT track_id, plate_text, valid, votes,
      ROUND(avg_confidence,2) AS conf, video_path
      FROM vehicle_plates ORDER BY avg_confidence DESC""", db))
#!/usr/bin/env python3
"""Indian number plate recognition: YOLO plate detector + fine-tuned TrOCR reader.

Usage:
    python anpr.py --video clip.mp4
    python anpr.py --video clip.mp4 --csv plates.csv --ocr-every 10
"""
import argparse, os, re
from collections import defaultdict, Counter

import cv2
import torch
import pandas as pd
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

HF_REPO = "subash1652007/indian-plate-ocr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- plate format helpers (Indian plates only) ----------
TO_DIGIT = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2'}
TO_LETTER = {v: k for k, v in TO_DIGIT.items()}
PATTERN = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]?[A-Z]{1,3}\d{4}$')   # e.g. TN09AB1234, DL3CCA1750
TEMPLATES = {
    8:  ["LLDLDDDD"],
    9:  ["LLDDLDDDD", "LLDLLDDDD"],
    10: ["LLDDLLDDDD", "LLDLLLDDDD"],
}
MIN_LEN, MAX_LEN = 8, 11      # matches the lengths the reader was trained on


def is_valid(t):
    return bool(PATTERN.match(t))


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
    if len(t) > 5:
        s = _snap(t[1:])          # drop one extra leading character
        if s:
            return s
    return t


def vote(reads):
    """reads = [(text, conf), ...] -> (consensus_text, n_reads_used, avg_conf)"""
    pool = [r for r in reads if is_valid(r[0])] or [r for r in reads if MIN_LEN <= len(r[0]) <= MAX_LEN]
    if not pool:
        return None, 0, 0.0
    L = Counter(len(t) for t, _ in pool).most_common(1)[0][0]
    same = [(t, c) for t, c in pool if len(t) == L]
    text = ''.join(Counter(t[i] for t, _ in same).most_common(1)[0][0] for i in range(L))
    return text, len(same), sum(c for _, c in same) / len(same)


# ---------- models ----------
class PlateReader:
    def __init__(self, trocr_dir):
        self.proc = TrOCRProcessor.from_pretrained(trocr_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_dir).to(DEVICE).eval()

    def __call__(self, crop):
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pv = self.proc(pil, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(pv, max_length=20, num_beams=8, num_return_sequences=5,
                                      output_scores=True, return_dict_in_generate=True)
        cands = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        probs = torch.exp(out.sequences_scores).tolist()
        texts = [fix(re.sub(r'[^A-Z0-9]', '', c.upper())) for c in cands]
        for t, p in zip(texts, probs):       # best candidate that follows the plate format
            if is_valid(t):
                return t, p
        return texts[0], probs[0]


def get_weights(args):
    if args.weights_dir:
        return args.weights_dir
    try:
        return snapshot_download(args.hf_repo, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        raise SystemExit(
            f"Could not download weights from '{args.hf_repo}': {e}\n"
            "If the repo is private, set HF_TOKEN or run `hf auth login` first.")

def merge_near_duplicates(rows, best_crop, max_diff=2):
    """Collapse low-vote reads that differ by <= max_diff characters from a better-supported read."""
    rows = sorted(rows, key=lambda r: (-r["votes"], -best_crop[r["track_id"]][0]))
    kept = []
    for r in rows:
        if r["votes"] <= 2 and any(
                len(r["plate_text"]) == len(k["plate_text"]) and
                sum(a != b for a, b in zip(r["plate_text"], k["plate_text"])) <= max_diff
                for k in kept):
            continue
        kept.append(r)
    return kept
# ---------- main pipeline ----------
def process_video(args, yolo, reader):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"Input: {args.video} | {w}x{h} | {fps:.0f} fps | {total} frames | device: {DEVICE}")
    if DEVICE == "cpu":
        print("No GPU found: this will be slow. Try --ocr-every 10 or higher.")

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or f"output_{stem}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    reads, best_crop, frame_no = defaultdict(list), {}, 0
    for res in yolo.track(source=args.video, stream=True, persist=True,
                          tracker="bytetrack.yaml", conf=args.det_conf, verbose=False):
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
                if frame_no % args.ocr_every == 0 and (x2 - x1) >= args.min_width:
                    text, conf = reader(crop)
                    if MIN_LEN <= len(text) <= MAX_LEN:
                        reads[tid].append((text, conf))
                label = vote(reads[tid])[0] if reads[tid] else None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{tid} {label or '...'}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(frame)
    writer.release()

    # one extra vote per track: read its largest crop
    for tid, (_, c) in best_crop.items():
        if c.shape[1] >= args.min_width:
            t, cf = reader(c)
            if MIN_LEN <= len(t) <= MAX_LEN:
                reads[tid].append((t, cf))

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    rows, seen = [], set()
    for tid, rs in reads.items():
        text, n, conf = vote(rs)
        if not text or text in seen:
            continue
        seen.add(text)
        if args.save_crops:
            cv2.imwrite(os.path.join(args.save_crops, f"{stem}_track{tid}.jpg"), best_crop[tid][1])
        rows.append({"track_id": tid, "plate_text": text, "valid_format": int(is_valid(text)),
                     "votes": n, "confidence": round(conf, 2)})
    rows = merge_near_duplicates(rows, best_crop)
    print(f"{frame_no} frames processed, {len(rows)} vehicles found -> {out_path}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Indian number plate recognition from video")
    ap.add_argument("--video", required=True, help="input video file")
    ap.add_argument("--out", help="annotated output video (default: output_<name>.mp4)")
    ap.add_argument("--csv", help="also save results to this CSV file")
    ap.add_argument("--save-crops", help="folder to save the best plate crop per vehicle (off by default)")
    ap.add_argument("--ocr-every", type=int, default=5, help="read plates every Nth frame")
    ap.add_argument("--min-width", type=int, default=60, help="ignore plates narrower than this (px)")
    ap.add_argument("--det-conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--hf-repo", default=HF_REPO, help="Hugging Face repo holding the weights")
    ap.add_argument("--weights-dir", help="local folder with yolo_best.pt and trocr_plate/ (skips download)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    wdir = get_weights(args)
    yolo = YOLO(os.path.join(wdir, "yolo_best.pt"))
    reader = PlateReader(os.path.join(wdir, "trocr_plate"))

    df = process_video(args, yolo, reader)
    if df.empty:
        print("No plates read.")
        return
    df = df.sort_values("confidence", ascending=False)
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
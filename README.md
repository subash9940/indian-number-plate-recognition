# Indian Number Plate Recognition

Detects vehicle number plates in video, tracks each vehicle, and reads the plate text. It works on **Indian plates only**.

**Pipeline:** YOLO plate detector → ByteTrack tracking → fine-tuned TrOCR reader → format correction and voting across frames.

Weights are hosted on Hugging Face: [`subash1652007/indian-plate-ocr`](https://huggingface.co/subash1652007/indian-plate-ocr). The script downloads them automatically on first run.

## Quick start

```bash
git clone https://github.com/subash9940/indian-number-plate-recognition.git
cd indian-number-plate-recognition
pip install -r requirements.txt
python anpr.py --video your_video.mp4
```

You get an annotated video (`output_<name>.mp4`) and a table of plates printed in the terminal.

A GPU is strongly recommended. On CPU it works but is slow, so raise `--ocr-every` (for example `10`). If your laptop has no GPU, use a free Google Colab GPU runtime and run the same four commands in notebook cells, prefixing each with `!`. Upload your video to the Colab session first.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--video` | required | input video |
| `--out` | `output_<name>.mp4` | annotated output video |
| `--csv` | off | also save results to a CSV |
| `--save-crops` | off | folder for the best plate crop per vehicle |
| `--ocr-every` | 5 | read plates every Nth frame |
| `--min-width` | 60 | ignore plates narrower than this (px) |
| `--det-conf` | 0.25 | detector confidence threshold |
| `--weights-dir` | none | use local weights instead of downloading |

Output columns: `track_id`, `plate_text`, `valid_format`, `votes`, `confidence`.

## How it works

1. **Detect** plates in each frame with a YOLO model and track them across frames with ByteTrack.
2. **Read** every Nth frame with a TrOCR-small model fine-tuned on Indian plate crops. It generates several candidates and keeps the best one that matches the Indian plate format.
3. **Correct** common character confusions by position (O/0, I/1, B/8, S/5, Z/2) and drop extra leading characters.
4. **Vote** across all reads of a vehicle, character by character, and add one extra read of its largest crop.

## Results

Exact-match accuracy on 190 held-out plate crops, using ground-truth boxes so only the reader is measured:

| Reader | Exact match |
|---|---|
| EasyOCR (pretrained, raw) | 22% (41/190) |
| EasyOCR + format post-processing | 31% (59/190) |
| Fine-tuned TrOCR (this repo) | 67% |

How much to trust these numbers:
- The validation set is small (190 crops), and frames from the same video were kept together in one split.
- The 67% is the best of 15 epochs, chosen on this same set, so it is optimistic. Individual epochs ranged from 44% to 67%.
- These are single-image numbers. Voting across video frames should help, but I did not measure the full pipeline on a labelled benchmark.
- I also ran the full pipeline on one short Indian clip. It found 6 vehicles. One plate that an early version misread (`HP15ZZZ`) came out correctly as `HP15D2222`. Two of the six entries were the same car, split into two tracks. This was a spot check, not a scored test.

## Limitations

- **Indian plates only.** The reader and the format rules assume the Indian pattern (for example `TN09AB1234`). Other countries' plates will be misread or rejected.
- **Plate length** is limited to 8–11 characters, matching the training data.
- **`valid_format = 1` does not mean the read is correct.** It only means the text fits the pattern. Check `votes`: reads with a single vote can be wrong or duplicates.
- **Duplicate vehicles:** when a car is partly hidden, the tracker can give it a new ID, so one car may appear twice.
- **Two-line plates, blur, low resolution and partly visible plates** are weak spots.
- The **detector** was trained on a separate, smaller plate dataset and was not fine-tuned on the Kaggle data. Missed or loose boxes will hurt the reading.

## Training data and models

- **Detector:** YOLO model trained on a Roboflow Universe "Indian License Plate Detection" dataset, under that dataset's own terms.
- **Reader:** [`microsoft/trocr-small-printed`](https://huggingface.co/microsoft/trocr-small-printed) fine-tuned on plate crops from the Kaggle dataset [Indian vehicle license plate dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) by Sai Sirisha N and collaborators. About 1,500 training images with plate text labels, 15 epochs, with colour, blur and box-jitter augmentation.
- The Kaggle data was split by source video to avoid near-duplicate frames leaking into validation.

## Privacy

License plates identify vehicles and, indirectly, people. By default the script saves only the annotated video and the text you ask for. It does not store crops or a database unless you pass `--save-crops`. Only run it on footage you have the right to process.

## Licence

- **Code:** AGPL-3.0, because the pipeline uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0.
- **TrOCR weights (`trocr_plate/` on Hugging Face):** trained on a dataset licensed **CC BY-NC-ND** (Attribution-NonCommercial-NoDerivatives) as listed on its Kaggle page. They are shared for **non-commercial, research and educational use only**. Do not use them commercially.
- **YOLO weights:** AGPL-3.0, and trained on a Roboflow dataset with its own terms.
- The images in the Kaggle dataset come from various sources and may belong to third parties. No warranty is given for any output.
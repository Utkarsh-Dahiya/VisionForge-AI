# Measuring plate-OCR accuracy (before vs after)

Accuracy can only be measured on **real labelled plate images with a real OCR engine**. The synthetic
generator (`generate_synthetic_plates.py`) draws Hershey-font text and is for plumbing tests only.

## 1. Build a labelled set
* 50+ photos or crops is a minimum for a rough signal; more is better. Include the hard cases (two-line
  two-wheeler plates, dusk, blur, angle, HSRP plates with the IND strip).
* `ground_truth.csv`:
  ```
  image_filename,ground_truth
  bike01.jpg,KL 01 BT 1650
  ```
  Spaces and hyphens are ignored. Only use numbers you have **verified by eye** on the image.

## 2. Run (PowerShell, project folder, venv active)
Whole photos (detector + OCR):
```powershell
python evaluate_ocr.py --images_dir .\photos --csv .\ground_truth.csv --model_path .\models\license_plate_detector.pt --pipelines legacy,revised --save_crops
```
Images that are already tight plate crops:
```powershell
python evaluate_ocr.py --images_dir .\crops --csv .\ground_truth.csv --mode crops --pipelines legacy,revised
```
Isolate each design decision (one change at a time against `revised`):
```powershell
python evaluate_ocr.py ... --pipelines revised,revised_early_exit,revised_confidence_rank,revised_legacy_variants
```
Try EasyOCR decoding options without editing code: `--readtext_opts decoder=beamsearch,beamWidth=5`.

## 3. Read the results
`ocr_eval_results\ocr_evaluation_report.csv` (one row per image per pipeline: ground truth, raw text,
processed text, status, latency, crop size) and `ocr_evaluation_metrics.json` (metrics + manifest with the
CSV hash so runs are comparable). Check `--save_crops` output first: if the crops are wrong
(wrong region, cut-off characters), fix detection before judging OCR.

Rules built in: every labelled row is in the denominator (missing files, failed detections and empty OCR
count as wrong); raw and processed text are recorded separately; all pipelines see the same images and
the same detection. "Plate found" means a box was found, not that the box is right (that needs box labels).
`--cache_ocr` speeds up multi-pipeline runs but makes latency figures invalid.
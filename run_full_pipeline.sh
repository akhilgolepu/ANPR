#!/bin/bash
# Full ANPR training pipeline: dataset prep → plate detection → OCR evaluation

set -e

echo "================================"
echo "FULL ANPR PIPELINE EXECUTION"
echo "================================"

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Step 1: Dataset prep (already done at: 2026-03-05 23:11)
echo ""
echo "[Step 1] Dataset already prepared"
echo "  Train: 3841 plates, Val: 480 plates, Test: 481 plates"

# Step 2: Plate detection training (in progress)
echo ""
echo "[Step 2] Plate detection training..."
echo "  Status: Ongoing (started at 23:23 UTC)"
echo "  Expected completion: ~2-3 hours"
echo "  Monitor: tail -f runs/plate_detection/yolov8s_6402/results.csv"

# Step 3: Wait for training to complete
waiting_for_training=true
while $waiting_for_training; do
  if [ -f "runs/plate_detection/yolov8s_6402/weights/best.pt" ]; then
    # Check if training still active
    if ps aux | grep -q "train_plate_detection.py" | grep -v grep >/dev/null 2>&1; then
      EPOCHS=$(tail -1 runs/plate_detection/yolov8s_6402/results.csv 2>/dev/null | cut -d',' -f1)
      echo "[$(date +%H:%M:%S)] Training: Epoch $EPOCHS / 120"
      sleep 60
    else
      waiting_for_training=false
    fi
  fi
done

echo ""
echo "[Step 2 Complete] Plate detection training finished"

# Step 4: OCR evaluation
echo ""
echo "[Step 3] Evaluating OCR recognition..."
conda run -n ml_workspace python scripts/train/train_ocr_model.py

# Step 5: Full pipeline evaluation
echo ""
echo "[Step 4] Full pipeline evaluation..."
conda run -n ml_workspace python scripts/evaluate_full_pipeline.py

echo ""
echo "================================"
echo "PIPELINE EXECUTION COMPLETE!"
echo "================================"
echo ""
echo "Reports saved to: reports/"
echo "  - reports/dataset_summary.json"
echo "  - reports/ocr_evaluation_report.json"
echo "  - reports/evaluation_report.json"
echo ""


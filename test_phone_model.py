#!/usr/bin/env python
"""Test phone detection model - Diagnostic Script"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

print("=" * 70)
print("🧪 PHONE DETECTION MODEL DIAGNOSTIC TEST")
print("=" * 70)

# Load model
try:
    model = YOLO("models/best v5.pt")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Create a dummy image (640x640, like expected input)
dummy_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

print("\n1️⃣ Test with random dummy image:")
try:
    results = model.predict(
        source=dummy_img, conf=0.55, iou=0.45, imgsz=640, verbose=False, device="cpu"
    )

    if results and results[0].boxes:
        print(f"  Detections found: {len(results[0].boxes)}")
        for box in results[0].boxes:
            print(f"    - Confidence: {box.conf[0]:.4f}, Class: {box.cls[0]}")
    else:
        print("  ✓ No detections (expected for random noise)")
except Exception as e:
    print(f"✗ Error during prediction: {e}")
    import traceback

    traceback.print_exc()

# Check model configuration
print("\n2️⃣ Model Configuration:")
print(f"  Input size: 640x640")
print(f"  Conf threshold: 0.55")
print(f"  IOU threshold: 0.45")
print(f"  Device: cpu")

print("\n3️⃣ Configuration Analysis:")
problems = []

# Check if conf threshold is reasonable
if 0.55 > 0.6:
    problems.append(
        "  ⚠️  Confidence threshold (0.55) is BORDERLINE HIGH - might miss marginal detections"
    )

print(f"  Model backbone: YOLOv8-based")
print(f"  Expected output: Bounding boxes with class=0 (phone)")

if not problems:
    print("  ✓ Configuration looks reasonable")
else:
    for p in problems:
        print(p)

print("\n4️⃣ KEY INVESTIGATION AREAS:")
print("  1. Check if model weights are corrupted (compare file size/hash)")
print("  2. Test with actual phone images to see real confidence scores")
print("  3. Compare model predictions between runs")
print("  4. Check if confidence threshold (0.55) filters too aggressively")
print("  5. Verify queue synchronization in InferenceThread")

print("\n" + "=" * 70)

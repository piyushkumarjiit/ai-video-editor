#!/usr/bin/env python3
"""
================================================================================
YOLO to TensorRT Bulk Converter (Self-Healing Edition)
================================================================================
DESCRIPTION:
    Automates the conversion of YOLOv8 PyTorch (.pt) models to TensorRT (.engine)
    format, optimized specifically for NVIDIA Pascal architecture (GTX 1080 Ti).
    
    Includes a "Self-Healing" mechanism to bypass PyTorch 2.6+ security blocks 
    (WeightsUnpickler) by auto-detecting and allowlisting required globals.

INPUTS:
    - .pt model files located in the 'models/' directory.
    - Approved model list (APPROVED_MODELS) to prevent unauthorized execution.

OUTPUTS:
    - .engine files moved to 'models/TensorModels/'.
    - Intermediate .onnx and .cache files are automatically purged.

REQUIREMENTS:
    - Hardware: NVIDIA GPU (1080 Ti 11GB recommended).
    - Software: PyTorch 2.6+, Ultralytics, TensorRT, ONNX, ONNX-Slim.
    - Environment: YOLO_SKIP_CHECK=True (enabled via script).

LIMITATIONS:
    - Fixed at FP32 (half=False) for Pascal stability.
    - Default imgsz=1088 for high-fidelity 4K analysis.
    - Not compatible with Re-ID models (OSNet) which lack YOLO dictionaries.
================================================================================
"""

import os
import shutil
import re
import torch
from ultralytics import YOLO

# --- Configuration ---
SOURCE_DIR = "models"
TARGET_DIR = os.path.join(SOURCE_DIR, "TensorModels")
USE_DYNAMIC = False  # Set to True for flexible batch/imgsz support

# Ensure target directory exists
os.makedirs(TARGET_DIR, exist_ok=True)
os.environ['YOLO_SKIP_CHECK'] = 'True'

# Approved list to allow for "Self-Healing" load
APPROVED_MODELS = ['yolov8x.pt', 'yolov8n-face.pt']

def self_healing_yolo_load(model_path):
    """
    Attempts to load a YOLO model and automatically registers missing 
    safe globals if PyTorch 2.6 security blocks them.
    """
    attempt_limit = 10
    attempts = 0
    
    while attempts < attempt_limit:
        try:
            return YOLO(model_path)
        except Exception as e:
            error_str = str(e)
            if "Unsupported global: GLOBAL" in error_str:
                # Extract the missing class name from the error message
                match = re.search(r"GLOBAL ([\w\.]+) was not an allowed global", error_str)
                if match:
                    missing_class_path = match.group(1)
                    print(f"🛠️ Auto-Discovery: Adding {missing_class_path} to safe globals...")
                    
                    parts = missing_class_path.split('.')
                    module_path = ".".join(parts[:-1])
                    class_name = parts[-1]
                    
                    try:
                        module = __import__(module_path, fromlist=[class_name])
                        klass = getattr(module, class_name)
                        torch.serialization.add_safe_globals([klass])
                        attempts += 1
                        continue 
                    except (ImportError, AttributeError) as import_err:
                        print(f"❌ Failed to auto-import {missing_class_path}: {import_err}")
                        break
            
            print(f"💥 Permanent error loading {model_path}: {e}")
            break
    return None

def register_safe_globals():
    """Register core classes. Self-healing handles the rest."""
    import collections
    import ultralytics.nn.modules.conv as ulconv
    import ultralytics.nn.modules.block as ulblock
    import ultralytics.nn.modules.head as ulhead
    import ultralytics.utils as ulutils
    import ultralytics.utils.loss as ulloss 
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel
    )

    def get_classes(module):
        import inspect
        return [
            obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if module.__name__ in (obj.__module__ or "")
        ]

    safe_classes = [
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel,
        *get_classes(ulutils), 
        *get_classes(ulloss),
        *get_classes(ulconv),
        *get_classes(ulblock),
        *get_classes(ulhead),
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.container.Sequential,
        collections.OrderedDict,
    ]
    torch.serialization.add_safe_globals(safe_classes)

def bulk_convert():
    """Scan SOURCE_DIR for .pt files and convert based on APPROVED_MODELS."""
    model_files = [f for f in os.listdir(SOURCE_DIR) 
                   if f.endswith('.pt') and os.path.isfile(os.path.join(SOURCE_DIR, f))]

    if not model_files:
        print(f"❓ No .pt models found in {SOURCE_DIR}")
        return

    for model_name in model_files:
        model_path = os.path.join(SOURCE_DIR, model_name)
        print(f"\n--- Processing: {model_name} ---")
        
        try:
            # Verify if model is in approved list
            if any(approved in model_name for approved in APPROVED_MODELS):
                model = self_healing_yolo_load(model_path)
            else:
                print(f"ℹ️ Skipping unapproved model {model_path}")
                continue

            if model:
                # Dynamic vs Static configuration
                export_params = {
                    "format": 'engine',
                    "device": 0,
                    "imgsz": 1088,
                    "half": False,      # FP32 stability for 1080 Ti
                    "workspace": 4,     # 4GB build limit
                    "simplify": True,
                    "dynamic": USE_DYNAMIC
                }

                if USE_DYNAMIC:
                    export_params["batch"] = 4
                    print("🌐 Exporting with Dynamic Shapes (Batch 1-4)...")
                else:
                    export_params["batch"] = 1
                    print("🔒 Exporting with Static Shape (Batch 1)...")

                model.export(**export_params)

                # Post-export cleanup and move
                engine_src = model_path.replace('.pt', '.engine')
                onnx_src = model_path.replace('.pt', '.onnx')
                
                if os.path.exists(engine_src):
                    target_path = os.path.join(TARGET_DIR, os.path.basename(engine_src))
                    shutil.move(engine_src, target_path)
                    print(f"✅ Success! Engine moved to: {target_path}")

                    # Cleanup intermediate artifacts
                    if os.path.exists(onnx_src):
                        os.remove(onnx_src)
                        print(f"🗑️ Cleaned up intermediate file: {onnx_src}")
                    
                    cache_file = model_path.replace('.pt', '.cache')
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                else:
                    print(f"❌ Export failed: {engine_src} was not created.")

        except Exception as e:
            if "osnet" in model_path.lower():
                print(f"ℹ️ Skipping Re-ID model {model_path}: Not a YOLO architecture.")
            else:
                print(f"💥 Critical error converting {model_path}: {e}")

if __name__ == "__main__":
    print(f"Starting auto-discovery conversion in '{SOURCE_DIR}'...")
    register_safe_globals()
    bulk_convert()
    print("\n🚀 All tasks complete.")
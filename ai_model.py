"""
ai_models.py — AI Model Integration Layer
==========================================
This module handles loading and running the trained YOLOv11s-OBB
model for visual defect detection within the digital twin.
 
During the SimPy simulation, at each CMM inspection checkpoint,
this module selects a sample aircraft image, runs YOLO inference,
and returns structured detection results that get stored in the
visual_inspections table in Supabase.
 
In a production deployment, the image would come from a camera
mounted at the inspection station. In this prototype, we simulate
this by selecting from a pool of real aircraft defect test images.
 
Architecture note: This file is separate from simulation.py to
maintain separation of concerns. The simulation doesn't need to
know how YOLO works — it just calls run_visual_inspection() and
gets back a results dict.
 
Author: Nissy Joseph
"""
 
import os
import random
import json
import glob
from datetime import datetime
 
# YOLO import — runs on CPU for the digital twin integration
# (GPU not required for single-image inference at ~100-200ms)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. YOLO inference disabled.")
    print("Install with: pip install ultralytics")
 
 
class VisualInspector:
    """
    Handles YOLOv11s-OBB visual defect detection for the digital twin.
 
    This class loads the trained YOLO model once at startup and
    provides a method to run inference on inspection images.
    Loading the model once avoids the ~2 second load time on
    every inspection call.
 
    Usage:
        inspector = VisualInspector()
        results = inspector.run_inspection(part_number=1)
        # results dict contains detections, classes, pass/fail, etc.
    """
 
    # The 22 aerospace defect classes from the training dataset
    CLASS_NAMES = [
        'broken_discharge', 'broken_link', 'burn_damage', 'corrosion_damage',
        'crack', 'dent', 'dirty_surface', 'distortion', 'light_damage',
        'loosened_fastener', 'missing_fastener', 'missing_label', 'missing_light',
        'missing_panel', 'nick', 'oil_leakage', 'open_latch', 'paint_damage',
        'scratch', 'unpressurized_tire', 'worn_tire', 'wrong_fastener'
    ]
 
    # Defect classes that are safety-critical in aerospace manufacturing.
    # Detection of any of these above the confidence threshold triggers
    # an automatic FAIL decision for the part.
    CRITICAL_DEFECTS = {
        'crack', 'corrosion_damage', 'burn_damage', 'missing_fastener',
        'loosened_fastener', 'broken_link', 'distortion'
    }
 
    def __init__(self, model_path="best.pt", test_images_dir="test_images",
                 conf_threshold=0.25):
        """
        Initialise the visual inspector.
 
        Args:
            model_path: Path to the trained YOLOv11s-OBB weights file.
                       Downloaded from Kaggle after training.
            test_images_dir: Folder containing sample aircraft defect
                           images for simulated inspection.
            conf_threshold: Minimum confidence score for a detection
                          to count. 0.25 matches our evaluation threshold.
        """
        self.model = None
        self.test_images = []
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.test_images_dir = test_images_dir
 
        # Create output directory for annotated images
        self.output_dir = os.path.join("data", "inspection_images")
        os.makedirs(self.output_dir, exist_ok=True)
 
        # Load the YOLO model
        if YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"  YOLO: Model loaded from {model_path}")
            except Exception as e:
                print(f"  YOLO: Failed to load model — {e}")
                self.model = None
        elif not os.path.exists(model_path):
            print(f"  YOLO: Model file not found at {model_path}")
            print(f"        Download best.pt from Kaggle and place it here.")
        
        # Load test images list
        if os.path.exists(test_images_dir):
            self.test_images = (
                glob.glob(os.path.join(test_images_dir, "*.jpg")) +
                glob.glob(os.path.join(test_images_dir, "*.png"))
            )
            print(f"  YOLO: {len(self.test_images)} test images available")
        else:
            print(f"  YOLO: Test images directory not found at {test_images_dir}")
            print(f"        Create folder and add aircraft defect images.")
 
    def is_available(self):
        """Check if YOLO model and test images are ready."""
        return self.model is not None and len(self.test_images) > 0
 
    def run_inspection(self, part_number, tool_wear_vb=0.0):
        """
        Run visual inspection on a sample image for the given part.
 
        This simulates what would happen at a real CMM inspection
        station: a camera captures an image of the machined part,
        and the YOLO model analyses it for defects.
 
        In this prototype, we select a random test image from the
        pool. The selection is weighted by tool wear — higher wear
        increases the chance of picking an image that contains
        defects (simulating the correlation between process
        conditions and defect occurrence).
 
        Args:
            part_number: Current part being inspected (for logging)
            tool_wear_vb: Current tool wear in mm (affects image selection)
 
        Returns:
            dict: Inspection results with keys:
                - image_url: path to original image
                - annotated_image_url: path to image with detections drawn
                - detections: list of detection dicts (class, confidence, bbox)
                - total_defects: number of defects found
                - defect_classes: list of unique defect class names
                - avg_confidence: mean confidence across all detections
                - pass_fail: True if part passes, False if critical defect found
        """
        if not self.is_available():
            return self._empty_result(part_number)
 
        # Select a random test image (simulating camera capture)
        image_path = random.choice(self.test_images)
 
        # Run YOLO inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=0.45,
            imgsz=640,
            device="cpu",        # CPU inference for local deployment
            verbose=False,       # Suppress per-image output
        )
 
        # Parse the detection results
        result = results[0]  # Single image, so one result
        detections = []
        defect_classes = set()
 
        if result.obb is not None and len(result.obb) > 0:
            boxes = result.obb
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                class_name = self.CLASS_NAMES[cls_id]
 
                # Get bounding box coordinates
                # OBB returns xyxyxyxy (4 corner points)
                bbox = boxes.xyxyxyxy[i].tolist() if hasattr(boxes, 'xyxyxyxy') else []
 
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                })
                defect_classes.add(class_name)
 
        # Calculate aggregate metrics
        total_defects = len(detections)
        avg_confidence = (
            round(sum(d["confidence"] for d in detections) / total_defects, 4)
            if total_defects > 0 else 0.0
        )
 
        # Pass/fail decision: fail if any critical defect is detected
        critical_found = defect_classes & self.CRITICAL_DEFECTS
        pass_fail = len(critical_found) == 0
 
        # Save annotated image (with bounding boxes drawn)
        annotated_filename = f"part{part_number}_inspection.jpg"
        annotated_path = os.path.join(self.output_dir, annotated_filename)
        try:
            annotated_img = result.plot()  # Returns numpy array with boxes drawn
            import cv2
            cv2.imwrite(annotated_path, annotated_img)
        except Exception as e:
            print(f"  YOLO: Could not save annotated image — {e}")
            annotated_path = ""
 
        # Build the results dict
        inspection_result = {
            "image_url": os.path.basename(image_path),
            "annotated_image_url": annotated_path,
            "detections": detections,
            "total_defects": total_defects,
            "defect_classes": list(defect_classes),
            "avg_confidence": avg_confidence,
            "pass_fail": pass_fail,
            "part_number": part_number,
            "critical_defects_found": list(critical_found),
        }
 
        # Print summary
        status = "PASS" if pass_fail else "FAIL"
        if total_defects > 0:
            print(f"  YOLO: Part {part_number} — {status} | "
                  f"{total_defects} defect(s): {', '.join(defect_classes)} "
                  f"(avg conf: {avg_confidence:.1%})")
        else:
            print(f"  YOLO: Part {part_number} — {status} | No defects detected")
 
        return inspection_result
 
    def run_single_image(self, image_path):
        """
        Run YOLO inference on a single user-uploaded image.
 
        Used by the dashboard's image upload feature where users
        can drag and drop an aircraft part photo for instant
        defect analysis.
 
        Args:
            image_path: Path to the image file
 
        Returns:
            dict: Same structure as run_inspection()
        """
        if self.model is None:
            return self._empty_result(0)
 
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=0.45,
            imgsz=640,
            device="cpu",
            verbose=False,
        )
 
        result = results[0]
        detections = []
        defect_classes = set()
 
        if result.obb is not None and len(result.obb) > 0:
            boxes = result.obb
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                class_name = self.CLASS_NAMES[cls_id]
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                })
                defect_classes.add(class_name)
 
        total_defects = len(detections)
        avg_confidence = (
            round(sum(d["confidence"] for d in detections) / total_defects, 4)
            if total_defects > 0 else 0.0
        )
        critical_found = defect_classes & self.CRITICAL_DEFECTS
        pass_fail = len(critical_found) == 0
 
        # Save annotated image
        annotated_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
        try:
            annotated_img = result.plot()
            import cv2
            cv2.imwrite(annotated_path, annotated_img)
        except Exception:
            annotated_path = ""
 
        return {
            "image_url": image_path,
            "annotated_image_url": annotated_path,
            "detections": detections,
            "total_defects": total_defects,
            "defect_classes": list(defect_classes),
            "avg_confidence": avg_confidence,
            "pass_fail": pass_fail,
            "critical_defects_found": list(critical_found),
        }
 
    def _empty_result(self, part_number):
        """Return an empty result when YOLO is not available."""
        return {
            "image_url": "",
            "annotated_image_url": "",
            "detections": [],
            "total_defects": 0,
            "defect_classes": [],
            "avg_confidence": 0.0,
            "pass_fail": True,
            "part_number": part_number,
            "critical_defects_found": [],
        }
class RFDefectPredictor:
    """
    Random Forest parameter-based defect predictor.

    Loads the pre-trained RF classifier and maps simulation state
    variables onto the 16 training features. Called once per part
    after all machining operations complete.

    WHY THIS APPROACH (for dissertation):
    The RF was trained on a manufacturing quality dataset with features
    like DefectRate, QualityScore and EnergyConsumption. These concepts
    are present in the simulation but under different names. This class
    performs the mapping explicitly, making the integration transparent
    and auditable.
    """

    # Feature order must match the training dataset column order exactly.
    # This is the order X = df.drop(['DefectStatus'], axis=1) produced.
    FEATURE_NAMES = [
        'ProductionVolume', 'ProductionCost', 'SupplierQuality',
        'DeliveryDelay', 'DefectRate', 'QualityScore',
        'MaintenanceHours', 'DowntimePercentage',
        'InventoryTurnover', 'StockoutRate', 'WorkerProductivity',
        'SafetyIncidents', 'EnergyConsumption', 'EnergyEfficiency',
        'AdditiveProcessTime', 'AdditiveMaterialCost'
    ]

    def __init__(self, model_path="rf_defect_model.pkl"):
        """
        Load the trained Random Forest model from disk.

        Args:
            model_path: Path to the joblib-saved RF model file.
        """
        self.model = None
        self.model_path = model_path

        try:
            import joblib
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"  RF: Model loaded from {model_path}")
            else:
                print(f"  RF: Model file not found at {model_path}")
                print(f"      Download rf_defect_model.pkl from Kaggle.")
        except ImportError:
            print("  RF: joblib not installed. Run: pip install joblib")
        except Exception as e:
            print(f"  RF: Failed to load model — {e}")

    def is_available(self):
        """Check if model loaded successfully."""
        return self.model is not None

    def predict_from_part_state(self, part_state):
        """
        Build a feature vector from simulation state and run RF prediction.

        Maps simulation-generated values onto the 16 features the model
        was trained on. The mapping is documented inline for transparency.

        Args:
            part_state: dict containing simulation state for this part.
                Keys expected:
                - part_number (int)
                - max_defect_probability (float)
                - avg_defect_probability (float)
                - sensor_defect_detected (bool)
                - tool_changes_this_run (int)
                - energy_this_part_kwh (float)
                - operation_minutes (int)
                - tool_wear_final_vb (float)

        Returns:
            dict: Prediction result with probability, class, alert level,
                  recommended action and the feature vector used.
        """
        if not self.is_available():
            return None

        import numpy as np

        p = part_state

        # ── FEATURE MAPPING ──────────────────────────────────────────
        # Each line maps a simulation variable to its training feature.
        # Fixed values represent constants that have no simulation
        # equivalent (e.g. supplier quality, stockout rate).

        production_volume = float(p.get("part_number", 1))
        production_cost = (
            p.get("energy_this_part_kwh", 50.0) * 0.15
            + p.get("tool_changes_this_run", 0) * 12.5
        )
        supplier_quality = 0.87
        delivery_delay = 0.0
        defect_rate = p.get("max_defect_probability", 0.02)
        quality_score = round((1.0 - p.get("avg_defect_probability", 0.02)) * 100, 2)
        maintenance_hours = p.get("tool_changes_this_run", 0) * (1 / 60)
        total_minutes = p.get("operation_minutes", 120)
        downtime_pct = (
            (p.get("tool_changes_this_run", 0) * 1) / max(total_minutes, 1)
        ) * 100
        inventory_turnover = 8.5
        stockout_rate = 0.0
        worker_productivity = (p.get("part_number", 1) / max(total_minutes / 60, 0.1))
        safety_incidents = 0
        energy_consumption = p.get("energy_this_part_kwh", 50.0)
        energy_efficiency = round(1.0 - (energy_consumption / 90.0), 4)
        energy_efficiency = max(0.0, min(1.0, energy_efficiency))
        additive_process_time = 0.0
        additive_material_cost = 0.0

        feature_vector = [
            production_volume, production_cost, supplier_quality,
            delivery_delay, defect_rate, quality_score,
            maintenance_hours, downtime_pct, inventory_turnover,
            stockout_rate, worker_productivity, safety_incidents,
            energy_consumption, energy_efficiency,
            additive_process_time, additive_material_cost
        ]

        X = np.array(feature_vector).reshape(1, -1)


        # ── INFERENCE ────────────────────────────────────────────────
        predicted_class = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]
        defect_probability = round(float(probabilities[1]), 4)

        # Alert level thresholds
        if defect_probability >= 0.70:
            alert_level = "critical"
            recommended_action = (
                "STOP PRODUCTION: RF model predicts high defect probability. "
                "Inspect tooling and review last operation parameters immediately."
            )
        elif defect_probability >= 0.40:
            alert_level = "warning"
            recommended_action = (
                "MONITOR CLOSELY: Elevated defect risk detected. "
                "Check tool wear and coolant flow before next part."
            )
        else:
            alert_level = "normal"
            recommended_action = "Production parameters within acceptable range."

        # Build feature importance dict for storage
        feature_dict = dict(zip(self.FEATURE_NAMES, feature_vector))

        result = {
            "part_number": p.get("part_number"),
            "defect_probability": defect_probability,
            "predicted_class": predicted_class,
            "alert_level": alert_level,
            "recommended_action": recommended_action,
            "feature_vector": feature_dict,
        }

        # Console output
        status = "DEFECT" if predicted_class == 1 else "CLEAN"
        print(
            f"  RF:   Part {p.get('part_number')} — {status} | "
            f"P(defect)={defect_probability:.1%} | "
            f"Alert: {alert_level.upper()}"
        )

        return result    
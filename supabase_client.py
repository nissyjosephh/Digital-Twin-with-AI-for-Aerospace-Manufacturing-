"""
supabase_client.py — Database Integration Layer
================================================
This module handles all communication between the digital twin
simulation and the Supabase PostgreSQL database. It provides
methods to store production runs, sensor readings, defect
predictions, and sustainability metrics.

Why a separate file: Separation of concerns. The simulation
(simulation.py) doesn't need to know about database connection
details. It just calls methods like store_sensor_reading().
If you later swapped Supabase for a different database, you'd
only change this file — nothing else.

Author: Nissy Joseph
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load credentials from .env file
load_dotenv()


class DigitalTwinDB:
    """
    Database interface for the CNC Digital Twin.

    This class manages all database operations. It's instantiated
    once per simulation run and handles:
    - Creating production run records
    - Batch-inserting sensor readings
    - Storing defect predictions
    - Storing sustainability metrics

    Batch insertion is important: instead of making 360 individual
    API calls (one per sensor reading), we buffer readings and
    insert them in batches of 50. This is faster and uses fewer
    API calls against Supabase's rate limits.
    """

    def __init__(self):
        """
        Initialise the Supabase client.

        Reads SUPABASE_URL and SUPABASE_KEY from environment
        variables loaded from the .env file.
        """
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "Missing SUPABASE_URL or SUPABASE_KEY in .env file. "
                "Copy them from Project Settings > API Keys in Supabase."
            )

        self.supabase: Client = create_client(url, key)

        # Cache the machine_id so we don't query it every time
        self.machine_id = None

        # Buffer for batch inserting sensor readings
        self.reading_buffer = []
        self.batch_size = 50

        # Current run ID
        self.run_id = None

    def get_machine_id(self, machine_name="CNC-5AX-001"):
        """
        Look up the machine UUID from the machines table.

        This queries the machines table for the row matching
        the machine name and returns its UUID. This UUID is used
        as the foreign key in all other tables.

        Args:
            machine_name: Name of the machine (must match a row
                         in the machines table)

        Returns:
            str: UUID of the machine
        """
        if self.machine_id:
            return self.machine_id

        response = (
            self.supabase.table("machines")
            .select("id")
            .eq("machine_name", machine_name)
            .execute()
        )

        if not response.data:
            raise ValueError(
                f"Machine '{machine_name}' not found in database. "
                f"Check the machines table in Supabase."
            )

        self.machine_id = response.data[0]["id"]
        return self.machine_id

    def create_production_run(self, config_params):
        """
        Create a new production run record.

        Called once at the start of each simulation. Stores the
        experiment parameters (tool change threshold, etc.) so
        you can compare runs later.

        Args:
            config_params: Dict with keys like material, num_parts,
                          tool_change_threshold, etc.

        Returns:
            str: UUID of the new production run
        """
        machine_id = self.get_machine_id()

        run_data = {
            "machine_id": machine_id,
            "material": config_params.get("material", "Ti-6Al-4V"),
            "operation_sequence": config_params.get(
                "operation_sequence", None
            ),
            "status": "running",
            "total_parts": 0,
            "defective_parts": 0,
            "tool_change_threshold": config_params.get(
                "tool_change_threshold", None
            ),
            "experiment_notes": config_params.get(
                "experiment_notes", None
            ),
        }

        response = (
            self.supabase.table("production_runs")
            .insert(run_data)
            .execute()
        )

        self.run_id = response.data[0]["id"]
        print(f"  DB: Production run created (ID: {self.run_id[:8]}...)")
        return self.run_id

    def store_sensor_reading(self, reading):
        """
        Buffer a sensor reading for batch insertion.

        Instead of inserting each reading individually (360 API
        calls per run), we buffer them and flush in batches.
        This is more efficient and respects API rate limits.

        Args:
            reading: Dict from SensorGenerator.generate_reading()
        """
        # Add the foreign keys and timestamp
        db_reading = {
            "run_id": self.run_id,
            "machine_id": self.machine_id,
            "timestamp": datetime.now().isoformat(),
            "sim_time_minutes": reading.get("sim_time_minutes"),
            "part_number": reading.get("part_number"),
            "operation": reading.get("operation"),
            "machine_state": reading.get("machine_state"),
            "vibration_rms": reading.get("vibration_rms"),
            "temperature_c": reading.get("temperature_c"),
            "cutting_force_n": reading.get("cutting_force_n"),
            "spindle_rpm": reading.get("spindle_rpm"),
            "feed_rate_mmmin": reading.get("feed_rate_mmmin"),
            "tool_wear_vb_mm": reading.get("tool_wear_vb_mm"),
            "coolant_flow_lmin": reading.get("coolant_flow_lmin"),
            "power_kw": reading.get("power_kw"),
            "defect_probability": reading.get("defect_probability"),
            "alert_level": reading.get("alert_level"),
            "cumulative_energy_kwh": reading.get(
                "cumulative_energy_kwh"
            ),
        }

        self.reading_buffer.append(db_reading)

        # Flush when buffer reaches batch size
        if len(self.reading_buffer) >= self.batch_size:
            self._flush_readings()

    def _flush_readings(self):
        """
        Insert buffered readings into the database.

        Called automatically when buffer is full, and manually
        at the end of the simulation to flush remaining readings.
        """
        if not self.reading_buffer:
            return

        try:
            self.supabase.table("sensor_readings").insert(
                self.reading_buffer
            ).execute()
            count = len(self.reading_buffer)
            self.reading_buffer = []
            print(f"  DB: Flushed {count} sensor readings")
        except Exception as e:
            print(f"  DB ERROR: Failed to flush readings: {e}")

    def update_run_complete(self, total_parts, defective_parts):
        """
        Update the production run record when simulation finishes.

        Marks the run as 'completed' and records final part counts.

        Args:
            total_parts: Total parts produced
            defective_parts: Number that failed quality check
        """
        # Flush any remaining buffered readings
        self._flush_readings()

        self.supabase.table("production_runs").update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "total_parts": total_parts,
            "defective_parts": defective_parts,
        }).eq("id", self.run_id).execute()

        print(f"  DB: Run marked complete "
              f"({total_parts} parts, {defective_parts} defective)")

    def store_sustainability_metrics(self, metrics):
        """
        Store sustainability metrics for this production run.

        Called once at the end of the simulation with the output
        from SensorGenerator.get_sustainability_metrics().

        Args:
            metrics: Dict with energy, CO2, material utilisation, etc.
        """
        db_metrics = {
            "run_id": self.run_id,
            "energy_kwh_total": metrics.get("energy_kwh_total"),
            "energy_kwh_per_part": metrics.get("energy_kwh_per_part"),
            "co2_kg_per_part": metrics.get("co2_kg_per_part"),
            "material_utilisation_pct": metrics.get(
                "material_utilisation_pct"
            ),
            "scrap_weight_kg": metrics.get("scrap_weight_kg"),
            "coolant_litres_used": metrics.get("coolant_litres_used"),
            "chip_to_part_ratio": metrics.get("chip_to_part_ratio"),
        }

        self.supabase.table("sustainability_metrics").insert(
            db_metrics
        ).execute()

        print(f"  DB: Sustainability metrics stored")

    def store_defect_prediction(self, prediction):
        """
        Store a Random Forest defect prediction.

        Called for each sensor reading that gets an RF prediction.
        Will be used when we integrate the RF model.

        Args:
            prediction: Dict with probability, class, importances
        """
        db_prediction = {
            "run_id": self.run_id,
            "sensor_reading_id": prediction.get("sensor_reading_id"),
            "defect_probability": prediction.get("defect_probability"),
            "predicted_class": prediction.get("predicted_class"),
            "alert_level": prediction.get("alert_level"),
            "feature_importances": prediction.get(
                "feature_importances"
            ),
            "recommended_action": prediction.get(
                "recommended_action"
            ),
        }

        self.supabase.table("defect_predictions").insert(
            db_prediction
        ).execute()

    def store_visual_inspection(self, inspection):
        """
        Store a YOLOv11 visual inspection result.

        Called at each CMM inspection checkpoint when YOLO
        processes an image.

        Args:
            inspection: Dict with detections, classes, confidence
        """
        db_inspection = {
            "run_id": self.run_id,
            "image_url": inspection.get("image_url"),
            "annotated_image_url": inspection.get(
                "annotated_image_url"
            ),
            "detections": inspection.get("detections"),
            "total_defects": inspection.get("total_defects", 0),
            "defect_classes": inspection.get("defect_classes", []),
            "avg_confidence": inspection.get("avg_confidence"),
            "pass_fail": inspection.get("pass_fail"),
        }

        self.supabase.table("visual_inspections").insert(
            db_inspection
        ).execute()

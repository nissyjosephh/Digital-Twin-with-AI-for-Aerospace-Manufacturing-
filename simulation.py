"""
simulation.py — SimPy CNC Digital Twin Simulation Engine
========================================================
This is the core discrete-event simulation. SimPy uses Python generators
(yield statements) to model processes that take simulated time.

When you call env.run(), SimPy executes all generator processes in
simulated time order — a 4-hour production shift completes in milliseconds
of real time.

YOLO Integration: At each CMM Inspection checkpoint, the simulation
triggers the YOLOv11s-OBB model to run visual defect detection on a
sample aircraft image. This simulates what a camera-based inspection
system would do at the end of each machining cycle. Results are stored
in the visual_inspections table in Supabase.
 
Author: Nissy Joseph
"""

import simpy
import pandas as pd
import os
import json
from datetime import datetime, timedelta

from config import (
    RANDOM_SEED, SIM_DURATION_MINS, SENSOR_INTERVAL,  
    NUM_PARTS, OPERATIONS, MACHINE_CONFIG,
    TOOL_WEAR_CONFIG, MachineState,
)
from sensors import SensorGenerator
from supabase_client import DigitalTwinDB
from ai_model import VisualInspector

class CNCDigitalTwin:
    """
    The main digital twin class orchestrating the entire simulation.

    This class creates the SimPy environment, manages the CNC machine
    as a SimPy Resource, and coordinates sensor generation, tool changes,
    and data collection across the multi-operation machining sequence.

    HOW SIMPY WORKS (for dissertation):
    SimPy is a process-based discrete-event simulation library. Processes
    are Python generator functions that yield events (like timeouts or
    resource requests). SimPy's event loop advances simulated time
    between yields, executing processes in chronological order. This
    means a 4-hour machining shift simulates in milliseconds, but the
    data produced has 1-minute resolution with realistic sensor physics.
    """

    def __init__(self):
        """Initialise the digital twin with all components."""

        # SimPy environment — manages simulated time
        # env.now gives current simulation time in minutes
        self.env = simpy.Environment()

        # The CNC machine as a SimPy Resource with capacity=1
        # This means only one operation can use it at a time (realistic)
        self.machine = simpy.Resource(self.env, capacity=1)

        # Sensor generator maintains persistent tool wear state
        self.sensor_gen = SensorGenerator(seed=RANDOM_SEED)

        # Current machine state (for dashboard/3D visualisation)
        self.current_state = MachineState.IDLE

        # Data collection — list of dicts, converted to DataFrame at end
        self.sensor_data = []
        self.event_log = []
        self.inspection_results = []  # stores YOLO inspection results

        # Production tracking
        self.parts_produced = 0
        self.parts_defective = 0
        self.tool_changes = 0

        # Simulation start timestamp (for realistic datetime values)
        self.sim_start_time = datetime.now()

        # Database connection
        try:
            self.db = DigitalTwinDB()
            self.db_connected = True
            print("Database: Connected to Supabase")
        except Exception as e:
            self.db = None
            self.db_connected = False
            print(f"Database: Running in CSV-only mode ({e})")

         # YOLO Visual Inspector — loads the trained model once at startup
        # The model file (best.pt) must be in the project root directory
        # Test images must be in the test_images/ directory
        self.inspector = VisualInspector(
            model_path="best.pt",
            test_images_dir="test_images",
            conf_threshold=0.25,
        )

    def run(self):
        """
        Execute the complete production simulation.

        This is the entry point. It:
        1. Registers the production process with SimPy
        2. Runs the simulation until all parts are complete
        3. Returns collected data as a pandas DataFrame

        Returns:
            tuple: (sensor_df, event_df, sustainability_metrics)
        """
        print(f"{'='*60}")
        print(f"  CNC DIGITAL TWIN SIMULATION")
        print(f"  Machine: {MACHINE_CONFIG['machine_name']}")
        print(f"  Parts to produce: {NUM_PARTS}")
        print(f"  Operations per part: {len(OPERATIONS)}")
        print(f"{'='*60}")
        print()

        # Create production run in database
        if self.db_connected:
            from config import WORKPIECE_CONFIG, TOOL_WEAR_CONFIG
            self.db.create_production_run({
                "material": WORKPIECE_CONFIG["material"],
                "operation_sequence": [op["name"] for op in OPERATIONS],
                "tool_change_threshold": TOOL_WEAR_CONFIG["tool_change_threshold_vb"],
                "experiment_notes": f"Threshold: {TOOL_WEAR_CONFIG['tool_change_threshold_vb']}mm, Parts: {NUM_PARTS}",
            })
        # Register the main production process with SimPy.
        # env.process() takes a generator function and schedules it.
        self.env.process(self._production_run())

        # Run until the production process completes.
        # SimPy advances simulated time between yield statements.
        self.env.run()

        # Convert collected data to pandas DataFrames
        sensor_df = pd.DataFrame(self.sensor_data)
        event_df = pd.DataFrame(self.event_log)

        # Calculate sustainability metrics
        sustainability = self.sensor_gen.get_sustainability_metrics(
            self.parts_produced, self.parts_defective
        )

        # Store sustainability metrics in database
        if self.db_connected:
            self.db.store_sustainability_metrics(sustainability)

        # Print summary
        self._print_summary(sensor_df, sustainability)

        return sensor_df, event_df, sustainability

    def _production_run(self):
        """
        SimPy generator process for the entire production run.

        This is a GENERATOR FUNCTION (contains yield statements).
        Each yield pauses this process for the specified simulated time,
        then SimPy resumes it. This is how SimPy models the passage
        of time without blocking real CPU time.
        """

        # ── MACHINE STARTUP ──────────────────────────────────────────
        self._log_event("PRODUCTION_START", "Production run initiated")
        self._set_state(MachineState.SETUP)

        # Machine warm-up takes 5 minutes (spindle warm-up, axis homing)
        yield self.env.timeout(5)
        self._log_event("SETUP_COMPLETE", "Machine warmed up, axes homed")

        # ── PRODUCE EACH PART ────────────────────────────────────────
        for part_num in range(1, NUM_PARTS + 1):
            print(f"\n--- PART {part_num}/{NUM_PARTS} ---")

            # Check if tool needs changing before starting new part
            if (self.sensor_gen.current_tool_wear_vb >=
                    TOOL_WEAR_CONFIG["tool_change_threshold_vb"]):
                yield from self._tool_change(part_num)

            # Run all operations for this part
            part_defective = False
            for operation in OPERATIONS:
                # Request the machine resource (queue if busy)
                with self.machine.request() as req:
                    yield req  # Wait until machine is available

                    # Check if this is the inspection operation
                    if operation["name"] == "inspection":
                        # Run YOLO visual inspection at CMM checkpoint
                        defect_detected = yield from self._execute_inspection(
                            operation, part_num
                        )
                    else:
                        # Normal machining operation
                        defect_detected = yield from self._execute_operation(
                            operation, part_num
                        )
 
                    if defect_detected:
                        part_defective = True

            # Part complete
            self.parts_produced += 1
            if part_defective:
                self.parts_defective += 1
                self._log_event(
                    "PART_DEFECTIVE",
                    f"Part {part_num} FAILED quality check"
                )
            else:
                self._log_event(
                    "PART_COMPLETE",
                    f"Part {part_num} PASSED quality check"
                )

            print(f"  Part {part_num}: {'FAIL' if part_defective else 'PASS'}"
                  f" | Tool wear: {self.sensor_gen.current_tool_wear_vb:.3f}mm")

        # ── PRODUCTION COMPLETE ──────────────────────────────────────
        self._set_state(MachineState.IDLE)
        self._log_event("PRODUCTION_END", "Production run completed")

        # Update database with final results
        if self.db_connected:
            self.db.update_run_complete(
                self.parts_produced, self.parts_defective
            )

    def _execute_operation(self, operation, part_num):
        """
        Execute a single machining operation, generating sensor data.

        This is where sensor readings are actually produced. For each
        minute of the operation, we generate one reading via the
        SensorGenerator class.

        Args:
            operation: Dict from OPERATIONS list
            part_num: Current part number (for logging)

        Yields:
            SimPy timeout events (one per sensor interval)

        Returns:
            bool: True if a defect was detected during this operation
        """
        op_name = operation["display_name"]
        duration = operation["duration_minutes"]
        defect_detected = False

        # Set machine state
        if operation["spindle_rpm"] > 0:
            self._set_state(MachineState.MACHINING)
        else:
            self._set_state(MachineState.IDLE)  # Inspection

        self._log_event(
            "OPERATION_START",
            f"Part {part_num}: {op_name} started"
        )

        print(f"  {op_name}: ", end="", flush=True)

        # Generate sensor readings for each minute of the operation
        for minute in range(duration):
            # Generate one sensor reading
            reading = self.sensor_gen.generate_reading(
                operation, elapsed_minutes_in_op=minute
            )

            # Add metadata
            reading["part_number"] = part_num
            reading["sim_time_minutes"] = round(self.env.now, 2)
            reading["real_timestamp"] = (
                self.sim_start_time
                + timedelta(minutes=self.env.now)
            ).isoformat()

            # Store the reading
            self.sensor_data.append(reading)
            
            # Send to database
            if self.db_connected:
                self.db.store_sensor_reading(reading)

            # Print progress indicator
            if reading["alert_level"] == "critical":
                print("!", end="", flush=True)
            elif reading["alert_level"] == "warning":
                print("*", end="", flush=True)
            else:
                print(".", end="", flush=True)

            # Check if defect probability exceeds threshold
            if reading["defect_probability"] >= 0.5:
                defect_detected = True

            # Check for mid-operation tool change need
            if (self.sensor_gen.current_tool_wear_vb >=
                    TOOL_WEAR_CONFIG["tool_change_threshold_vb"]
                    and operation["spindle_rpm"] > 0):
                self._log_event(
                    "TOOL_WEAR_ALERT",
                    f"VB={self.sensor_gen.current_tool_wear_vb:.3f}mm "
                    f"exceeds threshold during {op_name}"
                )

            # Advance simulated time by 1 minute.
            # This is where SimPy "pauses" this process and could
            # run other concurrent processes if they existed.
            yield self.env.timeout(1)

        print(f" [{duration}min]")

        self._log_event(
            "OPERATION_END",
            f"Part {part_num}: {op_name} completed"
        )

        return defect_detected

    def _execute_inspection(self, operation, part_num):
        """
        Execute CMM inspection with YOLO visual defect detection.
 
        This method runs during the 'inspection' operation in the
        machining sequence. It:
        1. Generates sensor readings for the inspection duration
           (probe measurements, no cutting)
        2. Triggers YOLOv11s-OBB inference on a sample image
        3. Stores the visual inspection results in Supabase
        4. Returns whether a defect was detected
 
        In a real factory, the CMM probe checks dimensions while
        a camera inspects the surface. Our simulation models both:
        - Dimensional check: via the sensor readings and defect probability
        - Visual check: via YOLO inference on aircraft defect images
 
        Args:
            operation: The inspection operation dict from OPERATIONS
            part_num: Current part number
 
        Yields:
            SimPy timeout events
 
        Returns:
            bool: True if defect detected (by sensor OR vision)
        """
        op_name = operation["display_name"]
        duration = operation["duration_minutes"]
        sensor_defect = False
 
        self._set_state(MachineState.IDLE)
        self._log_event(
            "INSPECTION_START",
            f"Part {part_num}: CMM Inspection + Visual AI started"
        )
 
        print(f"  {op_name}: ", end="", flush=True)
 
        # Generate sensor readings for inspection duration
        # (probe measurements, temperature stabilisation, etc.)
        for minute in range(duration):
            reading = self.sensor_gen.generate_reading(
                operation, elapsed_minutes_in_op=minute
            )
            reading["part_number"] = part_num
            reading["sim_time_minutes"] = round(self.env.now, 2)
            reading["real_timestamp"] = (
                self.sim_start_time
                + timedelta(minutes=self.env.now)
            ).isoformat()
 
            self.sensor_data.append(reading)
 
            if self.db_connected:
                self.db.store_sensor_reading(reading)
 
            print(".", end="", flush=True)
 
            if reading["defect_probability"] >= 0.5:
                sensor_defect = True
 
            yield self.env.timeout(1)
 
        print(f" [{duration}min]")
 
        # ── YOLO Visual Inspection ────────────────────────────────
        # Run the trained YOLOv11s-OBB model on a sample image.
        # This simulates what a camera-based inspection system
        # would do at the end of the machining cycle.
        vision_defect = False
 
        if self.inspector.is_available():
            inspection = self.inspector.run_inspection(
                part_number=part_num,
                tool_wear_vb=self.sensor_gen.current_tool_wear_vb,
            )
 
            # Store inspection results locally
            self.inspection_results.append(inspection)
 
            # Store in Supabase
            if self.db_connected:
                # Convert detections to JSON-serialisable format
                db_inspection = {
                    "image_url": inspection["image_url"],
                    "annotated_image_url": inspection["annotated_image_url"],
                    "detections": json.dumps(inspection["detections"]),
                    "total_defects": inspection["total_defects"],
                    "defect_classes": inspection["defect_classes"],
                    "avg_confidence": inspection["avg_confidence"],
                    "pass_fail": inspection["pass_fail"],
                }
                self.db.store_visual_inspection(db_inspection)
 
            # Vision-based defect detection
            if not inspection["pass_fail"]:
                vision_defect = True
                self._log_event(
                    "VISUAL_DEFECT_DETECTED",
                    f"Part {part_num}: YOLO detected {inspection['total_defects']} "
                    f"defect(s) — {', '.join(inspection['defect_classes'])}. "
                    f"Critical: {', '.join(inspection['critical_defects_found']) or 'None'}"
                )
            else:
                defect_msg = (
                    "No defects" if inspection["total_defects"] == 0
                    else f"{inspection['total_defects']} non-critical defect(s)"
                )
                self._log_event(
                    "VISUAL_INSPECTION_PASS",
                    f"Part {part_num}: YOLO — {defect_msg}"
                )
        else:
            self._log_event(
                "INSPECTION_END",
                f"Part {part_num}: CMM Inspection completed (YOLO unavailable)"
            )
 
        self._log_event(
            "INSPECTION_END",
            f"Part {part_num}: CMM Inspection completed"
        )
 
        # Defect detected if EITHER sensor analysis OR vision detects one
        return sensor_defect or vision_defect

    def _tool_change(self, part_num):
        """
        Perform an automatic tool change.

        In a real factory, the ATC (Automatic Tool Changer) swaps the
        worn tool for a new one from the magazine. Takes ~30 seconds
        on modern machines, but we model it as 1 minute for simplicity.
        """
        self._set_state(MachineState.TOOL_CHANGE)

        old_vb = self.sensor_gen.current_tool_wear_vb
        self.sensor_gen.perform_tool_change()
        self.tool_changes += 1

        self._log_event(
            "TOOL_CHANGE",
            f"Tool changed before Part {part_num}. "
            f"Old VB: {old_vb:.3f}mm -> New VB: 0.000mm"
        )

        print(f"  ** TOOL CHANGE ** (VB was {old_vb:.3f}mm)")

        # Tool change takes 1 minute
        yield self.env.timeout(1)

    # ─── HELPER METHODS ──────────────────────────────────────────────

    def _set_state(self, state):
        """Update machine state and log it."""
        self.current_state = state

    def _log_event(self, event_type, description):
        """Record an event in the event log."""
        self.event_log.append({
            "sim_time_minutes": round(self.env.now, 2),
            "real_timestamp": (
                self.sim_start_time
                + timedelta(minutes=self.env.now)
            ).isoformat(),
            "event_type": event_type,
            "description": description,
            "machine_state": self.current_state,
            "tool_wear_vb_mm": round(
                self.sensor_gen.current_tool_wear_vb, 4
            ),
        })

    def _print_summary(self, sensor_df, sustainability):
        """Print production run summary to console."""
        print(f"\n{'='*60}")
        print(f"  PRODUCTION RUN SUMMARY")
        print(f"{'='*60}")
        print(f"  Parts produced:   {self.parts_produced}")
        print(f"  Parts defective:  {self.parts_defective}")
        print(f"  First-pass yield: "
              f"{((self.parts_produced - self.parts_defective) / max(1, self.parts_produced)) * 100:.1f}%")
        print(f"  Tool changes:     {self.tool_changes}")
        print(f"  Total readings:   {len(sensor_df)}")
        print(f"  Sim duration:     {sensor_df['sim_time_minutes'].max():.0f} minutes")
        
        # YOLO inspection summary
        if self.inspection_results:
            total_inspections = len(self.inspection_results)
            total_defects_found = sum(
                r["total_defects"] for r in self.inspection_results
            )
            failed_inspections = sum(
                1 for r in self.inspection_results if not r["pass_fail"]
            )
            print(f"\n  --- Visual Inspection (YOLOv11s-OBB) ---")
            print(f"  Inspections run:  {total_inspections}")
            print(f"  Total defects:    {total_defects_found}")
            print(f"  Parts failed:     {failed_inspections}")
            print(f"  Visual yield:     "
                  f"{((total_inspections - failed_inspections) / max(1, total_inspections)) * 100:.1f}%")
 
        print(f"\n  --- Sustainability ---")
        print(f"  Energy total:     {sustainability['energy_kwh_total']:.1f} kWh")
        print(f"  Energy/part:      {sustainability['energy_kwh_per_part']:.1f} kWh")
        print(f"  CO2/part:         {sustainability['co2_kg_per_part']:.2f} kg")
        print(f"  Material util:    {sustainability['material_utilisation_pct']:.1f}%")
        print(f"  Scrap weight:     {sustainability['scrap_weight_kg']:.1f} kg")

        # Alert distribution
        if "alert_level" in sensor_df.columns:
            alerts = sensor_df["alert_level"].value_counts()
            print(f"\n  --- Alert Distribution ---")
            for level in ["normal", "warning", "critical"]:
                count = alerts.get(level, 0)
                pct = (count / len(sensor_df)) * 100
                print(f"  {level.upper():10s}: {count:4d} ({pct:.1f}%)")

        print(f"{'='*60}")


# =====================================================================
# ENTRY POINT — Run this file directly to execute the simulation
# =====================================================================

if __name__ == "__main__":
    # Create and run the digital twin
    twin = CNCDigitalTwin()
    sensor_df, event_df, sustainability = twin.run()

    # Save outputs to CSV
    os.makedirs("data", exist_ok=True)

    sensor_df.to_csv("data/sensor_readings.csv", index=False)
    event_df.to_csv("data/event_log.csv", index=False)

    print(f"\nData saved to data/ directory:")
    print(f"  sensor_readings.csv  ({len(sensor_df)} rows)")
    print(f"  event_log.csv        ({len(event_df)} rows)")
    print(f"\nOpen the CSVs to inspect the data the digital twin generated.")

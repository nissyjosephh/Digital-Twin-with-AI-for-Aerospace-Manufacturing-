"""
config.py — Digital Twin Configuration Constants
=================================================
Centralises ALL configurable parameters. When you need to adjust a sensor
range or add a new operation, you change ONE file.

Grounded in published data:
- Vibration thresholds: ISO 10816-3
- Tool wear limits: ISO 3685
- Cutting parameters: Machining Data Handbook for Ti-6Al-4V

Author: Nissy Joseph
Project: Smart Digital Twin with AI-Enhanced Defect Detection
University: Birmingham City University
"""

import numpy as np


# =====================================================================
# SIMULATION SETTINGS
# =====================================================================

RANDOM_SEED = 42          # Same seed as your ML experiments for consistency
SIM_DURATION_MINS = 240   # 4 hours = one typical production shift
SENSOR_INTERVAL = 60      # Readings every 60 sim-seconds (= 1 per minute)
                          # Gives 240 readings/run. Production would use 1-sec
NUM_PARTS = 3             # 3 parts per 4-hour shift (realistic for Ti aerospace)


# =====================================================================
# MACHINE SPECIFICATION
# =====================================================================
# Modelled on DMG MORI DMU 50 class — one of the most common 5-axis
# CNC machines in aerospace. Safran uses machines in this category.

MACHINE_CONFIG = {
    "machine_name": "CNC-5AX-001",
    "machine_type": "5-Axis CNC Milling Machine",
    "max_spindle_rpm": 20000,
    "max_power_kw": 50,
    "coolant_system": "flood",
}


# =====================================================================
# WORKPIECE SPECIFICATION
# =====================================================================
# Ti-6Al-4V (Grade 5 Titanium) — aerospace standard. Thermal conductivity
# of only 6.7 W/(m·K) means heat concentrates at cutting edge.

WORKPIECE_CONFIG = {
    "material": "Ti-6Al-4V",
    "hardness_hrc": 36,
    "thermal_conductivity": 6.7,   # W/(m·K)
    "raw_weight_kg": 12.5,
    "final_weight_kg": 4.2,
    "material_cost_per_kg": 35.0,  # GBP
}


# =====================================================================
# MACHINE STATES
# =====================================================================

class MachineState:
    """String constants for machine states. Used as class attributes
    rather than Python enum because they serialize directly to JSON
    and database strings without conversion."""
    IDLE = "idle"
    SETUP = "setup"
    MACHINING = "machining"
    TOOL_CHANGE = "tool_change"
    MAINTENANCE = "maintenance"
    BREAKDOWN = "breakdown"


# =====================================================================
# MACHINING OPERATIONS — The Multi-Operation Sequence
# =====================================================================
# This is your NOVEL CONTRIBUTION. Tool wear carries forward between
# operations, so defect risk increases throughout the sequence.
# Each operation has unique sensor profiles and defect signatures.

OPERATIONS = [
    {
        "name": "rough_milling",
        "display_name": "Rough Milling",
        "duration_minutes": 45,
        "spindle_rpm": 1200,                # Lower RPM, aggressive cuts
        "feed_rate_mmmin": 150,             # High feed for material removal
        "depth_of_cut_mm": 3.0,
        "tool_type": "dia_12mm_carbide_end_mill",
        "vibration_baseline": 1.2,          # mm/s — roughing vibrates more
        "temperature_baseline": 45,         # °C
        "cutting_force_baseline": 1200,     # N — high forces
        "coolant_flow_baseline": 30,        # L/min — max coolant
        "wear_rate_per_minute": 0.0015,     # mm VB/min — fastest wear
        "primary_defects": ["warping", "thermal_distortion", "surface_tear"],
    },
    {
        "name": "finish_milling",
        "display_name": "Finish Milling",
        "duration_minutes": 30,
        "spindle_rpm": 2000,                # Higher RPM for finer finish
        "feed_rate_mmmin": 80,
        "depth_of_cut_mm": 0.5,
        "tool_type": "dia_8mm_carbide_ball_nose",
        "vibration_baseline": 0.8,
        "temperature_baseline": 38,
        "cutting_force_baseline": 400,
        "coolant_flow_baseline": 25,
        "wear_rate_per_minute": 0.0008,
        "primary_defects": ["surface_roughness", "chatter_marks", "dimensional_error"],
    },
    {
        "name": "drilling",
        "display_name": "Drilling",
        "duration_minutes": 15,
        "spindle_rpm": 1800,
        "feed_rate_mmmin": 60,
        "depth_of_cut_mm": 8.0,
        "tool_type": "dia_6mm_carbide_drill",
        "vibration_baseline": 1.5,          # Drilling vibrates significantly
        "temperature_baseline": 50,         # Heat concentrates in hole
        "cutting_force_baseline": 800,
        "coolant_flow_baseline": 20,
        "wear_rate_per_minute": 0.0012,
        "primary_defects": ["burr_formation", "hole_position_error"],
    },
    {
        "name": "boring",
        "display_name": "Boring",
        "duration_minutes": 20,
        "spindle_rpm": 1500,
        "feed_rate_mmmin": 40,              # Very slow for precision
        "depth_of_cut_mm": 0.2,
        "tool_type": "dia_6mm_boring_bar",
        "vibration_baseline": 0.6,          # Must be low for precision
        "temperature_baseline": 35,
        "cutting_force_baseline": 300,
        "coolant_flow_baseline": 20,
        "wear_rate_per_minute": 0.0006,
        "primary_defects": ["dimensional_error", "roundness_error"],
    },
    {
        "name": "inspection",
        "display_name": "CMM Inspection",
        "duration_minutes": 10,
        "spindle_rpm": 0,                   # Machine idle
        "feed_rate_mmmin": 0,
        "depth_of_cut_mm": 0,
        "tool_type": "cmm_probe",
        "vibration_baseline": 0.2,          # Ambient only
        "temperature_baseline": 25,         # Cooling to ambient
        "cutting_force_baseline": 0,
        "coolant_flow_baseline": 0,
        "wear_rate_per_minute": 0.0,        # No cutting = no wear
        "primary_defects": [],
    },
]


# =====================================================================
# SENSOR THRESHOLDS (ISO-grounded)
# =====================================================================

SENSOR_THRESHOLDS = {
    "vibration_rms": {
        "normal_max": 2.0, "warning_max": 4.5, "critical_max": 7.0  # ISO 10816-3
    },
    "temperature_c": {
        "normal_max": 55, "warning_max": 70, "critical_max": 85
    },
    "cutting_force_n": {
        "normal_max": 1500, "warning_max": 2000, "critical_max": 2500
    },
    "tool_wear_vb_mm": {
        "normal_max": 0.20, "warning_max": 0.30, "critical_max": 0.60  # ISO 3685
    },
    "coolant_flow_lmin": {
        "normal_min": 10, "warning_min": 5, "critical_min": 2
    },
}


# =====================================================================
# SENSOR CORRELATION MATRIX
# =====================================================================
# Order: [vibration, temperature, cutting_force, tool_wear_effect, power]
# Cholesky decomposition generates correlated random values.
# These correlations reflect real physics: worn tool -> more force -> more heat

SENSOR_CORRELATION_MATRIX = np.array([
    [1.00, 0.65, 0.80, 0.85, 0.75],   # vibration
    [0.65, 1.00, 0.70, 0.70, 0.80],   # temperature
    [0.80, 0.70, 1.00, 0.90, 0.85],   # cutting_force
    [0.85, 0.70, 0.90, 1.00, 0.80],   # tool_wear_effect
    [0.75, 0.80, 0.85, 0.80, 1.00],   # power
])


# =====================================================================
# TOOL WEAR MODEL (Three-phase)
# =====================================================================

TOOL_WEAR_CONFIG = {
    "initial_vb": 0.0,
    "break_in_end_vb": 0.08,           # End of Phase I
    "accelerated_threshold_vb": 0.30,   # Start of Phase III (ISO 3685 limit)
    "steady_state_rate": 1.0,           # Phase II multiplier
    "accelerated_multiplier": 3.0,      # Phase III multiplier
    "tool_change_threshold_vb": 0.35,   # Auto tool change triggered here
}


# =====================================================================
# DEFECT PROBABILITY MODEL (Sigmoid)
# =====================================================================

DEFECT_MODEL_CONFIG = {
    "sigmoid_k": 20,                    # Steepness of S-curve
    "sigmoid_vb_midpoint": 0.30,        # 50% probability at ISO limit
    "base_defect_rate": 0.02,           # 2% even with new tools
    "coolant_failure_multiplier": 2.5,
    "vibration_excess_multiplier": 1.5,
}


# =====================================================================
# ANOMALY PROBABILITIES (per minute of operation)
# =====================================================================

ANOMALY_PROBABILITIES = {
    "coolant_pressure_drop": 0.02,      # 2%/min
    "vibration_spike": 0.03,            # 3%/min
    "tool_chipping": 0.005,             # 0.5%/min
}


# =====================================================================
# SUSTAINABILITY
# =====================================================================

SUSTAINABILITY_CONFIG = {
    "uk_grid_co2_kg_per_kwh": 0.233,    # UK grid carbon intensity 2024
}
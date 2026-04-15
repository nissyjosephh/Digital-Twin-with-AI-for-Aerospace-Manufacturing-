"""
sensors.py — Physics-Based Sensor Data Generator
=================================================
Generates realistic, correlated sensor data using Cholesky decomposition.
Maintains persistent state (tool wear) across operations — this is what
makes it a digital twin rather than a random number generator.

Author: Nissy Joseph
"""

import numpy as np
from scipy import linalg
from config import (
    SENSOR_CORRELATION_MATRIX, SENSOR_THRESHOLDS,
    ANOMALY_PROBABILITIES, TOOL_WEAR_CONFIG, DEFECT_MODEL_CONFIG,
)


class SensorGenerator:
    """
    Core sensor generation engine for the digital twin.

    WHY THIS CLASS EXISTS (for your dissertation):
    A digital twin must maintain STATE that evolves over time, mirroring
    the physical asset. This class tracks tool wear that accumulates across
    operations and between parts — exactly as a real cutting tool degrades.
    """

    def __init__(self, seed=42):
        """
        Initialise with fixed seed for reproducibility.

        The Cholesky decomposition of the correlation matrix is computed
        ONCE here (not every reading) — this is an optimisation to 
        mention in the implementation chapter.
        """
        # Dedicated RNG — doesn't affect global numpy random state
        self.rng = np.random.default_rng(seed)

        # Cholesky decomposition: L such that L @ L.T = correlation matrix
        # This is the mathematical core of correlated sensor generation
        self.cholesky_L = linalg.cholesky(
            SENSOR_CORRELATION_MATRIX, lower=True
        )

        # === PERSISTENT STATE (what makes this a "twin") ===
        self.current_tool_wear_vb = TOOL_WEAR_CONFIG["initial_vb"]
        self.cumulative_cutting_minutes = 0.0
        self.cumulative_energy_kwh = 0.0
        self.cumulative_coolant_litres = 0.0

        # Anomalies persist across readings until resolved
        self.active_anomalies = {
            "coolant_pressure_drop": False,
            "vibration_spike": False,
            "tool_chipping_occurred": False,
        }

    def generate_reading(self, operation, elapsed_minutes_in_op):
        """
        Generate ONE set of correlated sensor readings.

        Called once per sensor interval during simulation. Returns a dict
        that maps directly to the sensor_readings database table columns.

        The generation follows 8 steps, each grounded in physics:
        """

        # ── STEP 1: Update tool wear ────────────────────────────────
        # Tool wear ONLY increases during cutting (not inspection).
        # This carry-forward between operations is the key novelty.
        if operation["wear_rate_per_minute"] > 0:
            self.current_tool_wear_vb += self._calc_wear_increment(
                operation["wear_rate_per_minute"]
            )
            self.cumulative_cutting_minutes += 1.0

        # ── STEP 2: Generate correlated noise ────────────────────────
        # 5 independent N(0,1) values...
        z = self.rng.standard_normal(5)
        # ...transformed into correlated values via Cholesky multiplication.
        # After this: cn[0] (vibration) correlates with cn[2] (force) at r=0.80
        cn = self.cholesky_L @ z

        # ── STEP 3: Check for anomaly events ────────────────────────
        self._check_anomalies(operation)

        # Wear factor: normalises VB so ISO limit (0.3mm) = 1.0
        # This scales all sensor drift proportionally to wear
        wf = self.current_tool_wear_vb / TOOL_WEAR_CONFIG[
            "accelerated_threshold_vb"
        ]

        # ── STEP 4: Calculate each sensor value ─────────────────────
        # Pattern: baseline + wear_effect + correlated_noise + anomaly

        # VIBRATION (mm/s RMS)
        # Baseline varies by operation (roughing=1.2, finishing=0.8, etc.)
        # Wear increases vibration (dull tool cuts unevenly)
        # Anomaly spike adds 2.0 mm/s when active
        vibration = max(0.1,
            operation["vibration_baseline"]
            + wf * 1.5           # wear effect
            + cn[0] * 0.3        # correlated noise
            + (2.0 if self.active_anomalies["vibration_spike"] else 0)
        )

        # TEMPERATURE (°C) — Exponential rise model
        # T(t) = T_ambient + (T_baseline - T_ambient)(1 - e^(-t/tau))
        # tau=15 minutes means temp reaches 63% of max in 15 mins
        tau = 15.0
        rise = 1.0 - np.exp(-elapsed_minutes_in_op / tau)
        temperature = max(20.0,
            operation["temperature_baseline"] * rise
            + 25.0 * (1.0 - rise)    # starts at 25°C ambient
            + wf * 8.0               # worn tools generate more heat
            + cn[1] * 2.0            # correlated noise
        )

        # CUTTING FORCE (N)
        # Wear increases force up to 40% above baseline (dull tool = more force)
        cutting_force = max(0,
            operation["cutting_force_baseline"]
            + wf * operation["cutting_force_baseline"] * 0.4
            + cn[2] * operation["cutting_force_baseline"] * 0.08
        )

        # SPINDLE RPM — CNC-controlled, small jitter from load
        spindle_rpm = max(0, round(
            operation["spindle_rpm"]
            + self.rng.normal(0, 5)   # sigma=5 RPM gaussian jitter
            - wf * 10                  # slight droop under load
        ))

        # FEED RATE (mm/min) — adaptive override if force too high
        # Real CNC controllers reduce feed when force exceeds limits
        force_ratio = cutting_force / max(
            operation["cutting_force_baseline"], 1
        )
        adaptive_cut = max(0, (force_ratio - 1.2) * 0.1)
        feed_rate = max(0,
            operation["feed_rate_mmmin"] * (1.0 - adaptive_cut)
            + self.rng.normal(0, 2)
        )

        # COOLANT FLOW (L/min)
        coolant_flow = operation["coolant_flow_baseline"]
        if self.active_anomalies["coolant_pressure_drop"]:
            coolant_flow *= self.rng.uniform(0.3, 0.6)  # drops to 30-60%
        else:
            coolant_flow += self.rng.normal(0, 1.0)
        coolant_flow = max(0, coolant_flow)

        # POWER (kW) — proportional to force * speed
        if operation["spindle_rpm"] > 0:
            cp = (cutting_force * operation["feed_rate_mmmin"]) / 60000
            power = max(1.0, 5.0 + cp*15 + wf*5.0 + cn[4]*1.5)
        else:
            power = max(1.0, 3.0 + self.rng.normal(0, 0.2))

        # ── STEP 5: Update energy accumulator ────────────────────────
        self.cumulative_energy_kwh += power / 60.0  # kW * (1/60 hr)
        self.cumulative_coolant_litres += coolant_flow / 60.0  # L/min to L per minute interval

        # ── STEP 6: Defect probability (sigmoid) ────────────────────
        defect_prob = self._calc_defect_probability(
            vibration, coolant_flow, operation
        )

        # ── STEP 7: Alert level ─────────────────────────────────────
        if defect_prob >= 0.7:
            alert = "critical"
        elif defect_prob >= 0.3:
            alert = "warning"
        else:
            alert = "normal"

        # ── STEP 8: Return reading dict ─────────────────────────────
        return {
            "operation": operation["name"],
            "machine_state": (
                "machining" if operation["spindle_rpm"] > 0
                else "inspection"
            ),
            "vibration_rms": round(vibration, 4),
            "temperature_c": round(temperature, 2),
            "cutting_force_n": round(cutting_force, 2),
            "spindle_rpm": int(spindle_rpm),
            "feed_rate_mmmin": round(feed_rate, 2),
            "tool_wear_vb_mm": round(self.current_tool_wear_vb, 6),
            "coolant_flow_lmin": round(coolant_flow, 2),
            "power_kw": round(power, 3),
            "defect_probability": round(defect_prob, 4),
            "alert_level": alert,
            "cumulative_energy_kwh": round(
                self.cumulative_energy_kwh, 4
            ),
        }

    # ─── PRIVATE METHODS ─────────────────────────────────────────────

    def _calc_wear_increment(self, base_rate):
        """Three-phase tool wear model (ISO 3685 aligned)."""
        vb = self.current_tool_wear_vb
        cfg = TOOL_WEAR_CONFIG

        if vb < cfg["break_in_end_vb"]:
            mult = 2.0          # Phase I: break-in (rapid)
        elif vb < cfg["accelerated_threshold_vb"]:
            mult = 1.0          # Phase II: steady-state (linear)
        else:
            mult = cfg["accelerated_multiplier"]  # Phase III: accelerated

        # Small random variation (real wear isn't perfectly smooth)
        noise = max(0.5, min(1.5, self.rng.normal(1.0, 0.1)))
        return base_rate * mult * noise

    def _check_anomalies(self, operation):
        """Trigger/resolve random anomaly events."""
        if operation["spindle_rpm"] == 0:
            self.active_anomalies = {
                k: False for k in self.active_anomalies
            }
            return

        # Worn machines have more problems (realistic feedback loop)
        wm = 1.0 + (self.current_tool_wear_vb / 0.30)
        p = ANOMALY_PROBABILITIES

        # Coolant pressure drop
        if not self.active_anomalies["coolant_pressure_drop"]:
            if self.rng.random() < p["coolant_pressure_drop"] * wm:
                self.active_anomalies["coolant_pressure_drop"] = True
        elif self.rng.random() < 0.20:  # 20% chance of self-resolving
            self.active_anomalies["coolant_pressure_drop"] = False

        # Vibration spike
        if not self.active_anomalies["vibration_spike"]:
            if self.rng.random() < p["vibration_spike"] * wm:
                self.active_anomalies["vibration_spike"] = True
        elif self.rng.random() < 0.50:  # resolves quickly
            self.active_anomalies["vibration_spike"] = False

        # Tool chipping (PERMANENT until tool change)
        if not self.active_anomalies["tool_chipping_occurred"]:
            if self.rng.random() < p["tool_chipping"] * wm:
                self.active_anomalies["tool_chipping_occurred"] = True
                self.current_tool_wear_vb += 0.03  # sudden 0.03mm jump

    def _calc_defect_probability(self, vibration, coolant_flow, op):
        """Sigmoid: P = base + sigmoid(VB) * modifiers"""
        cfg = DEFECT_MODEL_CONFIG
        if op["spindle_rpm"] == 0:
            return 0.0

        vb = self.current_tool_wear_vb
        sigmoid = 1.0 / (
            1.0 + np.exp(-cfg["sigmoid_k"] * (vb - cfg["sigmoid_vb_midpoint"]))
        )
        prob = cfg["base_defect_rate"] + sigmoid * (
            1.0 - cfg["base_defect_rate"]
        )

        # Modifiers: bad conditions multiply defect risk
        if coolant_flow < SENSOR_THRESHOLDS["coolant_flow_lmin"]["warning_min"]:
            prob *= cfg["coolant_failure_multiplier"]
        if vibration > SENSOR_THRESHOLDS["vibration_rms"]["warning_max"]:
            prob *= cfg["vibration_excess_multiplier"]

        return min(1.0, max(0.0, prob))

    # ─── PUBLIC METHODS ──────────────────────────────────────────────

    def perform_tool_change(self):
        """Reset all wear state (new tool from magazine)."""
        self.current_tool_wear_vb = TOOL_WEAR_CONFIG["initial_vb"]
        self.cumulative_cutting_minutes = 0.0
        self.active_anomalies["tool_chipping_occurred"] = False

    def get_sustainability_metrics(self, num_parts, num_defective):
        """Calculate sustainability KPIs for the run."""
        from config import WORKPIECE_CONFIG, SUSTAINABILITY_CONFIG
        cw = WORKPIECE_CONFIG
        cs = SUSTAINABILITY_CONFIG

        mat_util = (cw["final_weight_kg"] / cw["raw_weight_kg"]) * 100
        scrap_per = cw["raw_weight_kg"] - cw["final_weight_kg"]
        total_scrap = (scrap_per * num_parts
                       + num_defective * cw["raw_weight_kg"])
        e_per_part = self.cumulative_energy_kwh / max(1, num_parts)

        return {
            "energy_kwh_total": round(self.cumulative_energy_kwh, 3),
            "energy_kwh_per_part": round(e_per_part, 3),
            "co2_kg_per_part": round(
                e_per_part * cs["uk_grid_co2_kg_per_kwh"], 4
            ),
            "material_utilisation_pct": round(mat_util, 1),
            "coolant_litres_used": round(self.cumulative_coolant_litres, 2),
            "scrap_weight_kg": round(total_scrap, 2),
            "chip_to_part_ratio": round(
                scrap_per / cw["final_weight_kg"], 2
            ),
        }

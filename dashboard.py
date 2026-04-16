"""
dashboard.py — Streamlit Dashboard for the CNC Digital Twin
============================================================
Read-only user dashboard for production teams (Nacelles, Machining,
Chemical Processing, etc.) at Safran. Displays sensor data, defect
predictions, sustainability metrics, and experiment comparisons.

Run with: streamlit run dashboard.py
Opens at: http://localhost:8501

Author: Nissy Joseph | Birmingham City University
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from supabase import create_client, Client

# Load Supabase credentials from .env
load_dotenv()


# =====================================================================
# PAGE CONFIGURATION (must be first Streamlit command)
# =====================================================================

st.set_page_config(
    page_title="Aerospace Manufacturing Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# DATABASE CONNECTION (cached so we don't reconnect on every interaction)
# =====================================================================

@st.cache_resource
def init_supabase() -> Client:
    """
    Initialise Supabase client. Cached as a resource so it persists
    across reruns. Streamlit reruns the entire script on every user
    interaction, so without caching we'd reconnect every time.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)


supabase = init_supabase()


# =====================================================================
# DATA FETCHING FUNCTIONS (cached for 30 seconds)
# =====================================================================

@st.cache_data(ttl=30)
def fetch_production_runs():
    """Fetch all production runs for the dropdown selector."""
    response = (
        supabase.table("production_runs")
        .select("*")
        .order("start_time", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data)


@st.cache_data(ttl=30)
def fetch_sensor_readings(run_id):
    """Fetch all sensor readings for a specific production run."""
    # Supabase returns max 1000 rows per query by default.
    # For larger runs; need to paginate, but 360 rows fits fine.
    response = (
        supabase.table("sensor_readings")
        .select("*")
        .eq("run_id", run_id)
        .order("sim_time_minutes")
        .limit(2000)
        .execute()
    )
    return pd.DataFrame(response.data)


@st.cache_data(ttl=30)
def fetch_sustainability_metrics(run_id=None):
    """Fetch sustainability metrics (single run or all runs)."""
    query = supabase.table("sustainability_metrics").select("*")
    if run_id:
        query = query.eq("run_id", run_id)
    response = query.execute()
    return pd.DataFrame(response.data)


@st.cache_data(ttl=30)
def fetch_all_runs_with_metrics():
    """
    Fetch all production runs joined with their sustainability metrics
    for the experiment comparison tab.
    """
    runs = fetch_production_runs()
    metrics = fetch_sustainability_metrics()

    if runs.empty or metrics.empty:
        return pd.DataFrame()

    # Merge runs with metrics on run_id
    combined = runs.merge(
        metrics, left_on="id", right_on="run_id", suffixes=("", "_metrics")
    )

    # Calculate first-pass yield
    combined["yield_pct"] = (
        (combined["total_parts"] - combined["defective_parts"])
        / combined["total_parts"].replace(0, 1) * 100
    )

    return combined


def create_defect_probability_chart(sensor_df: pd.DataFrame) -> go.Figure:
    """Create a detailed defect probability chart with risk bands."""
    fig = go.Figure()

    fig.add_hrect(y0=0.0, y1=0.3, fillcolor="rgba(16, 185, 129, 0.08)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.3, y1=0.7, fillcolor="rgba(245, 158, 11, 0.08)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(239, 68, 68, 0.08)",
                  line_width=0, layer="below")

    fig.add_hline(
        y=0.3,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Warning Threshold",
        annotation_position="right",
    )
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Critical Threshold",
        annotation_position="right",
    )

    fig.add_trace(go.Scatter(
        x=sensor_df["sim_time_minutes"],
        y=sensor_df["defect_probability"],
        mode="lines",
        line=dict(color="#3b82f6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.12)",
        customdata=sensor_df[["operation", "part_number"]],
        hovertemplate=(
            "Time: %{x:.0f} min<br>"
            "Defect Probability: %{y:.1%}<br>"
            "Operation: %{customdata[0]}<br>"
            "Part: %{customdata[1]}<extra></extra>"
        ),
        name="Defect Probability",
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            tickformat=".0%",
        ),
        xaxis=dict(title="Simulation Time (minutes)"),
        showlegend=False,
    )
    return fig


def build_full_run_comparison(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Create the detailed run comparison table used in Tab 4."""
    comparison_df = all_runs.copy()
    comparison_df["start_time"] = pd.to_datetime(comparison_df["start_time"])
    comparison_df = comparison_df.sort_values("start_time", ascending=False)
    comparison_df["run_label"] = [
        f"Run {idx + 1}" for idx in range(len(comparison_df))
    ]

    display_df = comparison_df[[
        "run_label", "start_time", "tool_change_threshold", "total_parts",
        "defective_parts", "yield_pct", "energy_kwh_per_part",
        "co2_kg_per_part", "status",
    ]].copy()

    display_df["start_time"] = display_df["start_time"].dt.strftime(
        "%d/%m/%Y %H:%M"
    )
    display_df.columns = [
        "Run", "Started", "Tool Threshold (mm)", "Total Parts",
        "Defective", "Yield %", "Energy/Part (kWh)",
        "CO₂/Part (kg)", "Status",
    ]
    return display_df


# =====================================================================
# SIDEBAR — RUN SELECTOR (shared across tabs)
# =====================================================================

st.sidebar.title("Digital Twin Dashboard")
st.sidebar.markdown("**Smart Manufacturing for Aerospace**")
st.sidebar.markdown("---")

runs_df = fetch_production_runs()

if runs_df.empty:
    st.sidebar.error("No production runs found in database.")
    st.error("No data available. Please run the simulation first.")
    st.stop()

# Build dropdown labels: "Run 1 — 2026-04-13 | Threshold: 0.35mm"
runs_df["label"] = runs_df.apply(
    lambda r: f"Run {r.name + 1} — "
              f"{pd.to_datetime(r['start_time']).strftime('%Y-%m-%d %H:%M')} | "
              f"Threshold: {r.get('tool_change_threshold', 'N/A')}mm",
    axis=1,
)

selected_label = st.sidebar.selectbox(
    "Select Production Run",
    runs_df["label"].tolist(),
)

selected_run = runs_df[runs_df["label"] == selected_label].iloc[0]
selected_run_id = selected_run["id"]

st.sidebar.markdown("---")
st.sidebar.markdown("**Run Information**")
st.sidebar.text(f"Material: {selected_run['material']}")
st.sidebar.text(f"Status: {selected_run['status']}")
st.sidebar.text(f"Total Parts: {selected_run['total_parts']}")
st.sidebar.text(f"Defective: {selected_run['defective_parts']}")
yield_pct = (
    (selected_run["total_parts"] - selected_run["defective_parts"])
    / max(selected_run["total_parts"], 1) * 100
)
st.sidebar.metric("First-Pass Yield", f"{yield_pct:.1f}%")


# =====================================================================
# MAIN DASHBOARD — TABS
# =====================================================================

st.title("Aerospace Manufacturing Digital Twin")
st.caption(
    "Real-time monitoring, AI-enhanced defect prediction, and "
    "sustainability analytics for 5-axis CNC machining of Ti-6Al-4V components."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Sensor Monitoring",
    "AI Defect Predictions",
    "Sustainability Metrics",
    "Experiment Comparison",
])


# =====================================================================
# TAB 1: LIVE SENSOR MONITORING
# =====================================================================

with tab1:
    st.header("Live Sensor Monitoring")
    st.caption(
        "Real-time sensor data from the simulated CNC-5AX-001 machine. "
        "Threshold lines are based on ISO 10816-3 (vibration) and ISO 3685 (tool wear)."
    )

    sensor_df = fetch_sensor_readings(selected_run_id)

    if sensor_df.empty:
        st.warning("No sensor data found for this run.")
    else:
        # ── KPI Row at the top ────────────────────────────────────────
        latest = sensor_df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            alert = latest["alert_level"]
            st.metric(
                "Current Alert Level",
                alert.upper()
            )

        with col2:
            max_cutting_defect = sensor_df[
                sensor_df["machine_state"] == "machining"
            ]["defect_probability"].max()
            st.metric(
                "Peak Defect Risk",
                f"{max_cutting_defect * 100:.1f}%",
                help="Maximum defect probability during cutting operations",
            )

        with col3:
            st.metric(
                "Tool Wear (VB)",
                f"{latest['tool_wear_vb_mm']:.3f} mm",
                delta=f"{(latest['tool_wear_vb_mm'] - sensor_df.iloc[0]['tool_wear_vb_mm']):.3f} mm",
                delta_color="inverse",
            )

        with col4:
            st.metric(
                "Cumulative Energy",
                f"{latest['cumulative_energy_kwh']:.1f} kWh"
            )

        st.markdown("---")

        # ── Alert Banner ──────────────────────────────────────────────
        max_defect = sensor_df[sensor_df["machine_state"] == "machining"]["defect_probability"].max()
        max_wear = sensor_df["tool_wear_vb_mm"].max()
        
        if selected_run["defective_parts"] > 0:
            st.error(
                f"PRODUCTION ALERT: {selected_run['defective_parts']} of "
                f"{selected_run['total_parts']} parts failed quality checks. "
                f"Maximum tool wear reached {max_wear:.3f}mm "
                f"(ISO 3685 limit: 0.30mm). "
                f"Peak defect probability: {max_defect*100:.1f}%."
            )
        elif max_wear > 0.20:
            st.warning(
                f"Tool wear reached {max_wear:.3f}mm. Consider lowering "
                f"tool change threshold to prevent future defects."
            )
        else:
            st.success(
                f"All {selected_run['total_parts']} parts passed quality checks. "
                f"Maximum tool wear: {max_wear:.3f}mm (within normal range)."
            )

        st.markdown("---")

        # ── Operation Defect Analysis ─────────────────────────────────
        st.subheader("Operation Defect Analysis")
        st.caption("Maximum defect probability per operation per part. Red indicates critical risk.")
        
        cutting_df = sensor_df[sensor_df["machine_state"] == "machining"].copy()
        if not cutting_df.empty:
            op_defect = (
                cutting_df.groupby(["part_number", "operation"])
                .agg(
                    max_defect_prob=("defect_probability", "max"),
                    max_tool_wear=("tool_wear_vb_mm", "max"),
                    avg_vibration=("vibration_rms", "mean"),
                )
                .reset_index()
            )
            op_defect["risk_level"] = op_defect["max_defect_prob"].apply(
                lambda x: "CRITICAL" if x >= 0.7 else ("WARNING" if x >= 0.3 else "NORMAL")
            )
            op_defect.columns = [
                "Part", "Operation", "Max Defect Prob", "Max Tool Wear (mm)",
                "Avg Vibration (mm/s)", "Risk Level",
            ]
            op_defect["Max Defect Prob"] = (op_defect["Max Defect Prob"] * 100).round(1).astype(str) + "%"
            op_defect["Max Tool Wear (mm)"] = op_defect["Max Tool Wear (mm)"].round(3)
            op_defect["Avg Vibration (mm/s)"] = op_defect["Avg Vibration (mm/s)"].round(2)
            
            st.dataframe(
                op_defect.style.apply(
                    lambda row: [
                        "background-color: rgba(239, 68, 68, 0.2)" if row["Risk Level"] == "CRITICAL"
                        else "background-color: rgba(245, 158, 11, 0.2)" if row["Risk Level"] == "WARNING"
                        else "" for _ in row
                    ], axis=1
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # ── Operation timeline ────────────────────────────────────────
        st.subheader("Operation Sequence")
        op_summary = (
            sensor_df.groupby("operation")["sim_time_minutes"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        op_summary.columns = ["Operation", "Start (min)", "End (min)", "Readings"]
        st.dataframe(op_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Sensor charts in 2x4 grid ────────────────────────────────
        st.subheader("Sensor Readings Over Time")

        # Define sensor configurations: (column, title, unit, thresholds)
        sensor_configs = [
            ("vibration_rms", "Vibration RMS", "mm/s",
             {"warning": 4.5, "critical": 7.0}),
            ("temperature_c", "Temperature", "°C",
             {"warning": 70, "critical": 85}),
            ("cutting_force_n", "Cutting Force", "N",
             {"warning": 2000, "critical": 2500}),
            ("tool_wear_vb_mm", "Tool Wear (VB)", "mm",
             {"warning": 0.30, "critical": 0.60}),
            ("spindle_rpm", "Spindle Speed", "RPM", None),
            ("feed_rate_mmmin", "Feed Rate", "mm/min", None),
            ("coolant_flow_lmin", "Coolant Flow", "L/min", None),
            ("power_kw", "Power Consumption", "kW",
             {"warning": 35, "critical": 45}),
        ]

        # Render in a 2-column grid
        for i in range(0, len(sensor_configs), 2):
            col_left, col_right = st.columns(2)

            for col, (col_name, title, unit, thresholds) in zip(
                [col_left, col_right], sensor_configs[i:i + 2]
            ):
                with col:
                    fig = px.line(
                        sensor_df,
                        x="sim_time_minutes",
                        y=col_name,
                        color="operation",
                        title=f"{title} ({unit})",
                        labels={
                            "sim_time_minutes": "Time (minutes)",
                            col_name: f"{title} ({unit})",
                        },
                    )

                    # Add threshold lines if defined
                    if thresholds:
                        if "warning" in thresholds:
                            fig.add_hline(
                                y=thresholds["warning"],
                                line_dash="dash",
                                line_color="orange",
                                annotation_text="Warning",
                                annotation_position="right",
                            )
                        if "critical" in thresholds:
                            fig.add_hline(
                                y=thresholds["critical"],
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Critical",
                                annotation_position="right",
                            )

                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Detailed defect probability chart
        st.subheader("Defect Probability Progression")
        st.caption(
            "Analytical defect risk trend across the full simulation, with "
            "warning and critical probability bands."
        )
        st.plotly_chart(
            create_defect_probability_chart(sensor_df),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Alert distribution pie chart ─────────────────────────────
        st.subheader("Alert Distribution")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            alert_counts = sensor_df["alert_level"].value_counts().reset_index()
            alert_counts.columns = ["Alert Level", "Count"]

            fig_pie = px.pie(
                alert_counts,
                values="Count",
                names="Alert Level",
                color="Alert Level",
                color_discrete_map={
                    "normal": "#10b981",
                    "warning": "#f59e0b",
                    "critical": "#ef4444",
                },
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("**Critical Alert Events**")
            critical_alerts = sensor_df[
                sensor_df["alert_level"] == "critical"
            ][[
                "sim_time_minutes", "operation", "part_number",
                "tool_wear_vb_mm", "defect_probability"
            ]].head(20)

            if critical_alerts.empty:
                st.info("No critical alerts in this run.")
            else:
                critical_alerts.columns = [
                    "Time (min)", "Operation", "Part #",
                    "Tool Wear (mm)", "Defect Prob."
                ]
                st.dataframe(
                    critical_alerts,
                    use_container_width=True,
                    hide_index=True,
                )
        # ── Closed-Loop Decision Support ─────────────────────────────
        st.markdown("---")
        st.subheader("Decision Support Recommendations")
        
        recommendations = []
        
        if max_wear > 0.30:
            recommendations.append(
                f"**Tool Change Required:** Final tool wear ({max_wear:.3f}mm) exceeded "
                f"ISO 3685 limit (0.30mm). Set tool_change_threshold to 0.20mm to trigger "
                f"automatic replacement before quality degrades. Estimated scrap savings: "
                f"£{selected_run['defective_parts'] * 12.5 * 35:.0f} per shift."
            )
        
        avg_temp = sensor_df[sensor_df["machine_state"] == "machining"]["temperature_c"].mean()
        if avg_temp > 55:
            recommendations.append(
                f"**Thermal Management:** Average cutting temperature ({avg_temp:.1f} C) exceeds "
                f"normal threshold (55 C). Consider reducing spindle RPM by 10% during rough "
                f"milling or increasing coolant flow rate to 35 L/min."
            )
        
        avg_vib = sensor_df[sensor_df["machine_state"] == "machining"]["vibration_rms"].mean()
        if avg_vib > 2.0:
            recommendations.append(
                f"**Vibration Control:** Average vibration ({avg_vib:.2f} mm/s) is elevated. "
                f"Reduce feed rate by 10% during rough milling to lower cutting forces "
                f"and extend tool life."
            )
        
        coolant_drops = sensor_df[sensor_df["coolant_flow_lmin"] < 10]
        if len(coolant_drops) > 0:
            recommendations.append(
                f"**Coolant System Check:** {len(coolant_drops)} readings showed coolant flow "
                f"below 10 L/min. Inspect coolant filter and pump pressure. Low coolant "
                f"increases defect probability by 2.5x."
            )
        
        if not recommendations:
            st.success(
                "No immediate actions required. All parameters within acceptable ranges."
            )
        else:
            for rec in recommendations:
                st.warning(rec)

# =====================================================================
# TAB 2: AI DEFECT PREDICTIONS (placeholder for now)
# =====================================================================

with tab2:
    st.header("AI Defect Predictions")
    st.caption(
        "Random Forest parameter-based predictions and YOLOv11 visual inspection results."
    )

    st.info(
        "This tab will be implemented after Random Forest and YOLOv11 "
        "are integrated into the simulation pipeline. Currently, the analytical "
        "sigmoid-based defect probability is shown in Tab 1."
    )

    st.markdown("### What this tab will show:")
    st.markdown("""
    **Random Forest Predictions:**
    - Comparison chart: analytical sigmoid vs ML prediction over time
    - Feature importance bar chart showing which sensors most influenced predictions
    - Alert history table with timestamps and recommended actions
    - Decision support panel with actionable insights

    **YOLOv11 Visual Inspection:**
    - Original and annotated inspection images side by side
    - Detected defect classes with bounding boxes and confidence scores
    - Pass/fail decisions per part
    - Most common defect types across all runs
    """)


# =====================================================================
# TAB 3: SUSTAINABILITY METRICS
# =====================================================================

with tab3:
    st.header("Sustainability Metrics")
    st.caption(
        "Energy consumption, carbon emissions, and material utilisation analytics. "
        "Aligned with Safran's NetZero objectives and UK grid carbon intensity (0.233 kg CO₂/kWh)."
    )

    metrics_df = fetch_sustainability_metrics(selected_run_id)

    if metrics_df.empty:
        st.warning("No sustainability data for this run.")
    else:
        m = metrics_df.iloc[0]

        # ── KPI Row ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Energy",
                f"{m['energy_kwh_total']:.1f} kWh",
                help="Total electrical energy consumed across all operations",
            )

        with col2:
            target_co2 = 15.0  # Hypothetical Safran target
            delta = m["co2_kg_per_part"] - target_co2
            st.metric(
                "CO₂ per Part",
                f"{m['co2_kg_per_part']:.2f} kg",
                delta=f"{delta:+.2f} vs target",
                delta_color="inverse",
                help=f"Target: {target_co2} kg CO₂/part",
            )

        with col3:
            st.metric(
                "Material Utilisation",
                f"{m['material_utilisation_pct']:.1f}%",
                help="Percentage of raw material that becomes the final part",
            )

        with col4:
            st.metric(
                "Coolant Used",
                f"{m['coolant_litres_used']:.0f} L"
                if pd.notna(m.get("coolant_litres_used")) else "N/A",
                help="Total cutting fluid consumption",
            )

        st.markdown("---")

        # ── Energy breakdown chart ────────────────────────────────────
        st.subheader("Energy Consumption by Operation")

        sensor_df = fetch_sensor_readings(selected_run_id)
        if not sensor_df.empty:
            energy_by_op = (
                sensor_df.groupby("operation")["power_kw"]
                .sum()
                .reset_index()
            )
            # Convert power readings (1 per minute) to kWh
            energy_by_op["energy_kwh"] = energy_by_op["power_kw"] / 60
            energy_by_op = energy_by_op.sort_values("energy_kwh", ascending=False)

            col_left, col_right = st.columns([2, 1])

            with col_left:
                fig_energy = px.bar(
                    energy_by_op,
                    x="operation",
                    y="energy_kwh",
                    title="Energy Consumption per Operation",
                    labels={
                        "operation": "Operation",
                        "energy_kwh": "Energy (kWh)",
                    },
                    color="energy_kwh",
                    color_continuous_scale="Reds",
                )
                fig_energy.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_energy, use_container_width=True)

            with col_right:
                st.markdown("**Operation Breakdown**")
                energy_by_op["CO2 (kg)"] = (
                    energy_by_op["energy_kwh"] * 0.233
                ).round(3)
                energy_by_op["energy_kwh"] = energy_by_op["energy_kwh"].round(2)
                energy_by_op.columns = ["Operation", "Power Sum", "Energy (kWh)", "CO₂ (kg)"]
                st.dataframe(
                    energy_by_op[["Operation", "Energy (kWh)", "CO₂ (kg)"]],
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

        # ── Material utilisation visualisation ───────────────────────
        st.subheader("Material Utilisation")

        col_a, col_b = st.columns(2)

        with col_a:
            # Donut chart: useful product vs scrap
            useful_pct = m["material_utilisation_pct"]
            scrap_pct = 100 - useful_pct

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Final Product", "Chips / Scrap"],
                values=[useful_pct, scrap_pct],
                hole=0.5,
                marker=dict(colors=["#10b981", "#94a3b8"]),
            )])
            fig_donut.update_layout(
                title="Material Utilisation Ratio",
                height=350,
                annotations=[dict(
                    text=f"{useful_pct:.1f}%",
                    x=0.5, y=0.5,
                    font_size=24,
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_b:
            st.markdown("**Material Flow Analysis**")
            st.markdown(f"""
            - **Raw material per billet:** 12.5 kg Ti-6Al-4V
            - **Final part weight:** 4.2 kg
            - **Material removed (chips):** 8.3 kg per part
            - **Total scrap this run:** {m['scrap_weight_kg']:.1f} kg
            - **Chip-to-part ratio:** {m['chip_to_part_ratio']:.2f}

            *Ti-6Al-4V costs approximately £35/kg. Reducing scrap directly
            translates to material cost savings and lower carbon footprint
            from primary titanium production.*
            """)

        st.markdown("---")

        # ── CO2 contextualisation ────────────────────────────────────
        st.subheader("Carbon Footprint Context")

        co2_per_part = m["co2_kg_per_part"]
        co2_total = m["energy_kwh_total"] * 0.233

        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.metric(
                "Total CO₂ This Run",
                f"{co2_total:.2f} kg",
            )
        with col_c2:
            # Equivalent in km driven by average car (0.12 kg CO2/km)
            km_equivalent = co2_total / 0.12
            st.metric(
                "Equivalent Car Travel",
                f"{km_equivalent:.0f} km",
                help="Based on average UK car emissions of 0.12 kg CO₂/km",
            )
        with col_c3:
            # Equivalent in trees needed to absorb (21 kg CO2/year per tree)
            trees = co2_total / 21
            st.metric(
                "Trees Needed (1 year)",
                f"{trees:.1f}",
                help="A mature tree absorbs ~21 kg CO₂ per year",
            )

        st.markdown("---")

        st.subheader("CO₂ Emissions per Part")
        all_runs_metrics_df = fetch_all_runs_with_metrics()

        if not all_runs_metrics_df.empty:
            co2_df = all_runs_metrics_df.copy()
            co2_df["start_time"] = pd.to_datetime(co2_df["start_time"])
            co2_df = co2_df.sort_values("start_time", ascending=True)
            co2_df["run_label"] = [
                f"Run {idx + 1}" for idx in range(len(co2_df))
            ]

            selected_mask = co2_df["id"] == selected_run_id
            bar_colors = [
                "#dc2626" if is_selected else "#fca5a5"
                for is_selected in selected_mask
            ]

            fig_co2 = go.Figure()
            fig_co2.add_trace(go.Bar(
                x=co2_df["run_label"],
                y=co2_df["co2_kg_per_part"],
                marker_color=bar_colors,
                customdata=co2_df[["tool_change_threshold", "yield_pct"]],
                text=co2_df["co2_kg_per_part"].round(2),
                textposition="outside",
                hovertemplate=(
                    "Run: %{x}<br>"
                    "CO₂ per Part: %{y:.2f} kg<br>"
                    "Tool Change Threshold: %{customdata[0]} mm<br>"
                    "Yield: %{customdata[1]:.1f}%<extra></extra>"
                ),
                name="CO₂ per Part",
            ))
            fig_co2.add_hline(
                y=target_co2,
                line_dash="dash",
                line_color="orange",
                annotation_text="Target",
                annotation_position="right",
            )
            fig_co2.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Production Run",
                yaxis_title="CO₂ per Part (kg)",
                showlegend=False,
            )
            st.plotly_chart(fig_co2, use_container_width=True)
        else:
            st.info("CO₂ comparison will appear once completed runs are available.")

        st.markdown("---")

        st.subheader("Sustainability Insights")
        insight_cols = st.columns(3)

        energy_saving = m["energy_kwh_per_part"] * 0.08
        if co2_per_part > target_co2:
            co2_message = (
                f"Current emissions are {co2_per_part:.2f} kg CO₂/part. "
                f"Achieving the {target_co2:.0f} kg target requires a "
                f"{((co2_per_part - target_co2) / co2_per_part) * 100:.0f}% reduction."
            )
        else:
            co2_message = (
                f"Current emissions are {co2_per_part:.2f} kg CO₂/part, which is "
                f"{((target_co2 - co2_per_part) / target_co2) * 100:.0f}% below the "
                f"{target_co2:.0f} kg target."
            )

        with insight_cols[0]:
            st.markdown("**Energy Reduction Opportunity**")
            st.write(
                "Reducing rough milling spindle RPM by 10% could save "
                f"approximately 8% energy per part ({energy_saving:.1f} kWh) "
                "with minimal cycle time impact."
            )

        with insight_cols[1]:
            st.markdown("**Material Utilisation**")
            st.write(
                f"At {m['material_utilisation_pct']:.1f}%, this run produced "
                f"{m['scrap_weight_kg']:.1f} kg of scrap. Optimising tool "
                "change timing could reduce defective parts and improve utilisation."
            )

        with insight_cols[2]:
            st.markdown("**Carbon Footprint**")
            st.write(co2_message)


# =====================================================================
# TAB 4: EXPERIMENT COMPARISON
# =====================================================================

with tab4:
    st.header("Experiment Comparison")
    st.caption(
        "Compare production runs across different parameter configurations to identify "
        "optimal manufacturing strategies. Use this to find the best balance between "
        "quality (yield), productivity, and sustainability."
    )

    all_runs = fetch_all_runs_with_metrics()

    if all_runs.empty:
        st.warning("No completed runs with sustainability data available.")
    else:
        st.subheader("Cross-Run Parameter Analysis")
        st.caption(
            "Compare production runs to identify optimal parameter combinations "
            "for quality, productivity, and sustainability."
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Total Runs", f"{len(all_runs)}")
        summary_col2.metric("Avg Yield", f"{all_runs['yield_pct'].mean():.1f}%")
        summary_col3.metric("Best Yield", f"{all_runs['yield_pct'].max():.1f}%")
        summary_col4.metric(
            "Avg Energy/Part",
            f"{all_runs['energy_kwh_per_part'].mean():.1f} kWh",
        )

        st.markdown("---")

        st.subheader("Full Run Comparison")
        st.dataframe(
            build_full_run_comparison(all_runs),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

    if not all_runs.empty and len(all_runs) == 1:
        st.info(
            f"Only 1 production run found in the database. "
            f"Run more simulations with different parameters "
            f"(e.g., change `tool_change_threshold_vb` in config.py to 0.20, 0.25, etc.) "
            f"to enable comparison analysis."
        )

        st.markdown("### Current Run Summary")
        run = all_runs.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Threshold", f"{run.get('tool_change_threshold', 'N/A')} mm")
        col2.metric("Yield", f"{run['yield_pct']:.1f}%")
        col3.metric("Energy/Part", f"{run['energy_kwh_per_part']:.1f} kWh")
        col4.metric("CO₂/Part", f"{run['co2_kg_per_part']:.2f} kg")
    elif not all_runs.empty:
        # ── Summary table of all runs ────────────────────────────────
        st.subheader("All Production Runs")

        display_df = all_runs[[
            "start_time", "tool_change_threshold", "total_parts",
            "defective_parts", "yield_pct", "energy_kwh_per_part",
            "co2_kg_per_part", "material_utilisation_pct",
        ]].copy()

        display_df["start_time"] = pd.to_datetime(
            display_df["start_time"]
        ).dt.strftime("%Y-%m-%d %H:%M")
        display_df.columns = [
            "Date", "Threshold (mm)", "Parts", "Defective",
            "Yield (%)", "kWh/Part", "CO₂/Part (kg)", "Material Util (%)"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # ── Comparison charts ────────────────────────────────────────
        st.subheader("Parameter Impact Analysis")

        col_left, col_right = st.columns(2)

        with col_left:
            # Yield vs threshold
            fig_yield = px.scatter(
                all_runs,
                x="tool_change_threshold",
                y="yield_pct",
                size="total_parts",
                title="First-Pass Yield vs Tool Change Threshold",
                labels={
                    "tool_change_threshold": "Tool Change Threshold (mm)",
                    "yield_pct": "First-Pass Yield (%)",
                },
                trendline="ols" if len(all_runs) >= 3 else None,
            )
            fig_yield.update_traces(marker=dict(color="#10b981"))
            fig_yield.update_layout(height=400)
            st.plotly_chart(fig_yield, use_container_width=True)

        with col_right:
            # Energy vs threshold
            fig_energy = px.scatter(
                all_runs,
                x="tool_change_threshold",
                y="energy_kwh_per_part",
                size="total_parts",
                title="Energy per Part vs Tool Change Threshold",
                labels={
                    "tool_change_threshold": "Tool Change Threshold (mm)",
                    "energy_kwh_per_part": "Energy per Part (kWh)",
                },
                trendline="ols" if len(all_runs) >= 3 else None,
            )
            fig_energy.update_traces(marker=dict(color="#f59e0b"))
            fig_energy.update_layout(height=400)
            st.plotly_chart(fig_energy, use_container_width=True)

        st.markdown("---")

        # ── Pareto front: yield vs CO2 ───────────────────────────────
        st.subheader("Quality vs Sustainability Trade-off")

        fig_pareto = px.scatter(
            all_runs,
            x="co2_kg_per_part",
            y="yield_pct",
            size="total_parts",
            color="tool_change_threshold",
            hover_data=["tool_change_threshold", "defective_parts"],
            title="Pareto Front: First-Pass Yield vs CO₂ Emissions",
            labels={
                "co2_kg_per_part": "CO₂ per Part (kg)",
                "yield_pct": "First-Pass Yield (%)",
                "tool_change_threshold": "Threshold (mm)",
            },
            color_continuous_scale="Viridis",
        )
        fig_pareto.update_layout(height=500)
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown("""
        **How to read this chart:** Each point is one production run. The
        ideal configuration is in the **top-left** (high yield, low CO₂).
        Runs in the bottom-right represent inefficient configurations.
        Larger circles indicate runs that produced more parts.
        """)

        st.markdown("---")

        # ── Decision support ────────────────────────────────────────
        st.subheader("Recommendation")

        best_run = all_runs.loc[all_runs["yield_pct"].idxmax()]
        worst_run = all_runs.loc[all_runs["yield_pct"].idxmin()]

        scrap_savings = (
            (worst_run["defective_parts"] - best_run["defective_parts"])
            * 12.5 * 35  # 12.5 kg titanium at £35/kg
        )

        st.success(f"""
        **Recommended configuration:** Tool change threshold of
        **{best_run['tool_change_threshold']} mm** achieves the highest
        first-pass yield of **{best_run['yield_pct']:.1f}%**.

        **Estimated savings vs worst configuration:**
        - **{worst_run['defective_parts'] - best_run['defective_parts']} fewer defective parts**
        - **£{scrap_savings:.0f} saved in material costs per shift**
        - **{(worst_run['co2_kg_per_part'] - best_run['co2_kg_per_part']) * best_run['total_parts']:.1f} kg CO₂ avoided per shift**
        """)


# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")
st.caption(
    "Smart Digital Twin with AI-Enhanced Defect Detection for Aerospace Manufacturing | "
    "Nissy Joseph | Birmingham City University | Industry Partner: Safran"
)

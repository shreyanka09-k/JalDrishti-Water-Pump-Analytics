import streamlit as st
import pandas as pd
import numpy as np
import random

# Try ML model; if sklearn isn't installed, the dashboard still opens
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="JalDrishti - Water Pump Analytics",
    page_icon="💧",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("💧 JalDrishti")
st.subheader("Rural Drinking Water Pump Failure Prediction & Maintenance Prioritization")

st.markdown(
    """
    **JalDrishti** is an AI/ML-based prototype designed to identify
    rural drinking-water pumps that may require preventive maintenance.
    
    The dashboard uses synthetic telemetry data for demonstration.
    """
)

st.divider()


# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------
@st.cache_data
def generate_data(n=400, seed=42):

    np.random.seed(seed)
    random.seed(seed)

    districts = [
        "North Bengal",
        "South Bengal",
        "Bankura",
        "Purulia",
        "Murshidabad",
        "Nadia",
        "Hooghly",
        "Malda"
    ]

    pump_types = [
        "Submersible",
        "Centrifugal",
        "Solar Pump"
    ]

    data = []

    for i in range(n):

        district = random.choice(districts)
        pump_type = random.choice(pump_types)

        recency = np.random.randint(1, 61)
        tariff_freq = np.random.randint(0, 13)
        om_reserve = np.random.randint(5000, 100001)

        energy_ratio = round(
            np.random.uniform(25, 90), 2
        )

        runtime = round(
            np.random.uniform(2, 16), 2
        )

        priority = np.random.choice(
            ["Normal", "High", "Critical"],
            p=[0.55, 0.30, 0.15]
        )

        aquifer_drop = round(
            np.random.uniform(0, 5), 2
        )

        # Synthetic risk formula
        risk_score = 0

        if recency > 35:
            risk_score += 0.25

        if tariff_freq >= 8:
            risk_score += 0.20

        if om_reserve < 25000:
            risk_score += 0.20

        if energy_ratio < 45:
            risk_score += 0.25

        if runtime > 12:
            risk_score += 0.10

        failure = 1 if risk_score >= 0.50 else 0

        data.append([
            district,
            pump_type,
            recency,
            tariff_freq,
            om_reserve,
            energy_ratio,
            runtime,
            priority,
            aquifer_drop,
            failure
        ])

    columns = [
        "District",
        "Pump_Type",
        "Recency_Days",
        "Tariff_Freq_6M",
        "OM_Reserve_INR",
        "Energy_Discharge_Ratio",
        "Daily_Runtime_Hrs",
        "Utility_Priority",
        "Aquifer_Drop_m",
        "Failure"
    ]

    return pd.DataFrame(data, columns=columns)


df = generate_data(400)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

aquifer_threshold = st.sidebar.slider(
    "Aquifer stress threshold (m)",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.1
)

mechanic_budget = st.sidebar.slider(
    "Mechanic van budget",
    min_value=1,
    max_value=20,
    value=5
)


# ---------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------
features = [
    "Recency_Days",
    "Tariff_Freq_6M",
    "OM_Reserve_INR",
    "Energy_Discharge_Ratio",
    "Daily_Runtime_Hrs"
]

X = df[features]
y = df["Failure"]

if SKLEARN_AVAILABLE:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    df["ML_Risk"] = model.predict_proba(X)[:, 1]

else:

    # Fallback risk calculation so the app still works
    df["ML_Risk"] = (
        (df["Recency_Days"] / 60) * 0.20
        + (df["Tariff_Freq_6M"] / 12) * 0.15
        + (1 - df["OM_Reserve_INR"] / 100000) * 0.20
        + (1 - df["Energy_Discharge_Ratio"] / 90) * 0.30
        + (df["Daily_Runtime_Hrs"] / 16) * 0.15
    )

    accuracy = None


# ---------------------------------------------------------
# RISK CLASSIFICATION
# ---------------------------------------------------------
def classify_risk(row):

    risk = row["ML_Risk"]

    if risk >= 0.70:
        return "🔴 Critical"
    elif risk >= 0.45:
        return "🟠 High"
    elif risk >= 0.25:
        return "🟡 Medium"
    else:
        return "🟢 Low"


df["Risk_Level"] = df.apply(
    classify_risk,
    axis=1
)


# ---------------------------------------------------------
# HYDROGEOLOGICAL GUARDRAIL
# ---------------------------------------------------------
def apply_guardrail(row):

    if (
        row["ML_Risk"] >= 0.45
        and row["Aquifer_Drop_m"] >= aquifer_threshold
    ):
        return "💧 Hydrogeological Stress"

    return row["Risk_Level"]


df["Final_Status"] = df.apply(
    apply_guardrail,
    axis=1
)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
total_pumps = len(df)

high_risk = len(
    df[df["Risk_Level"].isin(["🔴 Critical", "🟠 High"])]
)

hydro_stress = len(
    df[df["Final_Status"] == "💧 Hydrogeological Stress"]
)

avg_energy_ratio = df["Energy_Discharge_Ratio"].mean()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🚰 Pumps Monitored",
        total_pumps
    )

with col2:
    st.metric(
        "⚠️ High/Critical Risk",
        high_risk
    )

with col3:
    st.metric(
        "💧 Hydro Stress",
        hydro_stress
    )

with col4:
    st.metric(
        "⚡ Avg Energy/Discharge",
        f"{avg_energy_ratio:.1f}"
    )


if accuracy is not None:
    st.info(
        f"Random Forest validation accuracy on synthetic test data: "
        f"**{accuracy * 100:.1f}%**"
    )
else:
    st.warning(
        "Scikit-learn was not available, so the dashboard is using "
        "the built-in risk scoring fallback."
    )


st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Maintenance Priority",
    "📊 Analytics",
    "💧 Guardrail",
    "📋 Raw Data"
])


# ---------------------------------------------------------
# TAB 1 - MAINTENANCE
# ---------------------------------------------------------
with tab1:

    st.header("🚨 Maintenance Priority")

    priority_df = df.sort_values(
        "ML_Risk",
        ascending=False
    ).head(mechanic_budget)

    display_cols = [
        "District",
        "Pump_Type",
        "Recency_Days",
        "Energy_Discharge_Ratio",
        "Daily_Runtime_Hrs",
        "Aquifer_Drop_m",
        "Risk_Level",
        "Final_Status"
    ]

    st.dataframe(
        priority_df[display_cols],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Top {mechanic_budget} pumps recommended for mechanic dispatch."
    )


# ---------------------------------------------------------
# TAB 2 - ANALYTICS
# ---------------------------------------------------------
with tab2:

    st.header("📊 Pump Analytics")

    st.subheader("Risk Distribution")

    risk_counts = (
        df["Risk_Level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Number of Pumps")
    )

    st.bar_chart(
        risk_counts.set_index("Risk Level")
    )

    st.subheader("Energy-to-Discharge Ratio")

    st.line_chart(
        df[
            [
                "Energy_Discharge_Ratio"
            ]
        ].head(100)
    )

    st.subheader("Daily Runtime")

    st.line_chart(
        df[
            [
                "Daily_Runtime_Hrs"
            ]
        ].head(100)
    )


# ---------------------------------------------------------
# TAB 3 - GUARDRAIL
# ---------------------------------------------------------
with tab3:

    st.header("💧 Hydrogeological Guardrail")

    st.write(
        """
        The guardrail prevents the system from treating every high-risk
        pump as a purely mechanical problem.
        
        When predicted mechanical risk is high **and** aquifer decline
        exceeds the selected threshold, the dashboard flags the location
        for hydrogeological attention.
        """
    )

    st.metric(
        "Current Aquifer Stress Threshold",
        f"{aquifer_threshold:.1f} m"
    )

    stress_df = df[
        df["Final_Status"] == "💧 Hydrogeological Stress"
    ]

    st.write(
        f"Locations currently flagged for hydrogeological stress: "
        f"**{len(stress_df)}**"
    )

    if len(stress_df) > 0:

        st.dataframe(
            stress_df[
                [
                    "District",
                    "Pump_Type",
                    "Aquifer_Drop_m",
                    "ML_Risk",
                    "Final_Status"
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------
# TAB 4 - RAW DATA
# ---------------------------------------------------------
with tab4:

    st.header("📋 Synthetic Telemetry Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Dataset as CSV",
        data=csv,
        file_name="jaldrishti_synthetic_telemetry.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "JalDrishti | AI/ML prototype for rural drinking-water pump "
    "maintenance prioritization | Synthetic data for academic demonstration"
)

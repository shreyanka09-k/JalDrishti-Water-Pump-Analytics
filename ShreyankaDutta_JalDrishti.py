import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="JalDrishti - Pump Breakdown Early Warning",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0E4D92;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F7FAFC;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #0E4D92;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Synthetic Data Engine
# ---------------------------------------------------------
@st.cache_data
def generate_synthetic_telemetry_data(n_samples=400, seed=42):
    np.random.seed(seed)
    
    district_clusters = ["Cluster A - North Plains", "Cluster B - Southern Hills", 
                          "Cluster C - Eastern Delta", "Cluster D - Western Arid Zone"]
    
    pump_types = ["Solar Submersible 5HP", "Grid Electric 10HP", "Dual Solar-Grid 7.5HP", "Handpump Motor Assist 3HP"]
    
    data = []
    
    for i in range(1, n_samples + 1):
        pump_id = f"PUMP-JJM-{i:04d}"
        gp_name = f"Gram Panchayat #{100 + (i % 45)}"
        cluster = np.random.choice(district_clusters)
        pump_type = np.random.choice(pump_types)
        
        # Telemetry & Socio-Economic Features
        # Recency: Days since last operator preventive check-in (1 to 120 days)
        recency_raw = np.random.randint(1, 120)
        
        # Frequency: Tariff collection drives in past 6 months (0 to 6)
        frequency_raw = np.random.randint(0, 7)
        
        # Monetary: GP O&M Fund Balance in INR (₹ 500 to ₹ 45,000)
        monetary_raw = np.random.randint(500, 45000)
        
        # Energy-to-Discharge Ratio: Liters pumped per kWh (20 to 180 L/kWh)
        # Normal pump ~120-160 L/kWh. Damaged impeller/friction ~30-70 L/kWh.
        energy_discharge_ratio_raw = np.round(np.random.normal(loc=110, scale=35), 2)
        energy_discharge_ratio_raw = np.clip(energy_discharge_ratio_raw, 15.0, 200.0)
        
        # Daily Pumping Runtime (Hours/Day)
        runtime_hours = np.round(np.random.uniform(2.5, 12.0), 1)
        
        # Public Utility Priority (1 = School/PHC/High Density, 0 = Standard)
        public_priority = np.random.choice([0, 1], p=[0.65, 0.35])
        
        # CGWB Regional Aquifer Depth Drop (meters relative to 3-year baseline)
        # Values > 2.0m indicate severe hydrogeological stress/heatwave
        aquifer_drop_m = np.round(np.random.exponential(scale=0.9), 2)
        
        # Create Messy Raw Strings for Realistic Cleaning Pipeline
        recency_str = f"{recency_raw} days" if np.random.rand() > 0.1 else str(recency_raw)
        monetary_str = f"₹ {monetary_raw:,}" if np.random.rand() > 0.15 else f"{monetary_raw}"
        ratio_str = f"{energy_discharge_ratio_raw} L/kWh" if np.random.rand() > 0.1 else str(energy_discharge_ratio_raw)
        
        # Base failure probability generation (Ground Truth Target for ML)
        # High recency, low frequency, low monetary, and low energy/discharge ratio drive breakdown risk
        risk_score = (
            (recency_raw / 120.0) * 0.35 +
            (1.0 - (frequency_raw / 6.0)) * 0.20 +
            (1.0 - min(monetary_raw / 30000.0, 1.0)) * 0.15 +
            (1.0 - min(energy_discharge_ratio_raw / 140.0, 1.0)) * 0.30
        )
        
        breakdown_risk_binary = 1 if (risk_score > 0.58 or energy_discharge_ratio_raw < 45.0) else 0
        
        data.append({
            "Pump_ID": pump_id,
            "Gram_Panchayat": gp_name,
            "Cluster": cluster,
            "Pump_Type": pump_type,
            "Recency_Raw": recency_str,
            "Tariff_Freq_6M": frequency_raw,
            "OM_Reserve_Raw": monetary_str,
            "Energy_Discharge_Ratio_Raw": ratio_str,
            "Daily_Runtime_Hrs": runtime_hours,
            "Public_Utility_Priority": public_priority,
            "CGWB_Aquifer_Drop_m": aquifer_drop_m,
            "Ground_Truth_Breakdown": breakdown_risk_binary
        })
        
    return pd.DataFrame(data)

# ---------------------------------------------------------
# Data Hygiene & Feature Engineering Pipeline
# ---------------------------------------------------------
def clean_and_preprocess_data(df):
    cleaned_df = df.copy()
    
    # 1. Clean Recency (Days)
    cleaned_df["Recency_Days"] = (
        cleaned_df["Recency_Raw"]
        .astype(str)
        .str.extract(r'(\d+)')
        .astype(float)
        .fillna(cleaned_df["Recency_Raw"].str.extract(r'(\d+)').astype(float).median())
    )
    
    # 2. Clean Monetary (O&M Reserve INR)
    cleaned_df["OM_Reserve_INR"] = (
        cleaned_df["OM_Reserve_Raw"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )
    cleaned_df["OM_Reserve_INR"] = cleaned_df["OM_Reserve_INR"].fillna(cleaned_df["OM_Reserve_INR"].median())
    
    # 3. Clean Energy-to-Discharge Ratio (L/kWh)
    cleaned_df["Energy_Discharge_Ratio"] = (
        cleaned_df["Energy_Discharge_Ratio_Raw"]
        .astype(str)
        .str.replace("L/kWh", "", regex=False)
        .str.strip()
        .astype(float)
    )
    cleaned_df["Energy_Discharge_Ratio"] = cleaned_df["Energy_Discharge_Ratio"].fillna(cleaned_df["Energy_Discharge_Ratio"].median())
    
    return cleaned_df

# ---------------------------------------------------------
# ML Model Training Pipeline
# ---------------------------------------------------------
def train_breakdown_classifier(df):
    feature_cols = [
        "Recency_Days", "Tariff_Freq_6M", "OM_Reserve_INR", 
        "Energy_Discharge_Ratio", "Daily_Runtime_Hrs"
    ]
    X = df[feature_cols]
    y = df["Ground_Truth_Breakdown"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Return trained model, test metrics, and predictions on entire dataset
    df_result = df.copy()
    df_result["ML_Breakdown_Prob"] = clf.predict_proba(X)[:, 1]
    df_result["ML_Predicted_Risk"] = (df_result["ML_Breakdown_Prob"] >= 0.5).astype(int)
    
    return clf, acc, df_result

# ---------------------------------------------------------
# Ethical Guardrail Override Engine
# ---------------------------------------------------------
def apply_ethical_guardrail(df, threshold_m=2.0):
    """
    ETHICAL GUARDRAIL LOGIC:
    If a pump shows high breakdown/failure risk BUT the regional groundwater 
    aquifer drop exceeds the threshold (>2.0 meters), it indicates drought/hydrogeological 
    stress rather than technician negligence or equipment friction failure.
    
    Overrides label to: "Hydrogeological Stress (Borewell Deepening / Tanker Relief)"
    """
    results = []
    
    for _, row in df.iterrows():
        is_ml_risk = row["ML_Predicted_Risk"] == 1
        aquifer_drop = row["CGWB_Aquifer_Drop_m"]
        
        if is_ml_risk and aquifer_drop >= threshold_m:
            final_status = "Hydrogeological Stress (Tanker/Borewell Deepening)"
            action_code = "OVERRIDE_AQUIFER"
            color = "#DD6B20" # Orange
        elif is_ml_risk:
            final_status = "High Mechanical Risk (Mechanic Van Required)"
            action_code = "DISPATCH_MECHANIC"
            color = "#E53E3E" # Red
        else:
            final_status = "Optimal / Normal Operation"
            action_code = "NORMAL"
            color = "#38A169" # Green
            
        results.append({
            "Final_Classification": final_status,
            "Action_Code": action_code,
            "Status_Color": color
        })
        
    res_df = pd.DataFrame(results)
    return pd.concat([df, res_df], axis=1)

# ---------------------------------------------------------
# Streamlit Main Execution Flow
# ---------------------------------------------------------
def main():
    # Application Title Banner
    st.markdown('<p class="main-header">💧 JalDrishti: Rural Water Scheme Pump Breakdown Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Jal Jeevan Mission - Predictive Maintenance & Capacity Dispatch Decision Support Platform</p>', unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.header("🕹️ Control Panel & Parameters")
    
    sample_size = st.sidebar.slider("Synthetic Log Dataset Size", min_value=100, max_value=1000, value=400, step=50)
    
    st.sidebar.subheader("🛡️ Ethical Guardrail Settings")
    aquifer_threshold = st.sidebar.slider(
        "CGWB Aquifer Depth Drop Override Threshold (Meters)", 
        min_value=1.0, max_value=4.0, value=2.0, step=0.1,
        help="If groundwater drops beyond this threshold, ML failure predictions are overridden to route emergency water relief instead of penalizing local technicians."
    )
    
    st.sidebar.subheader("🚐 Capacity Dispatch Limits")
    max_mechanic_vans = st.sidebar.slider(
        "Monthly Mechanic Van Dispatch Budget", 
        min_value=5, max_value=50, value=20, step=1,
        help="Cap field dispatches based on monthly operational budget limitations."
    )
    
    # Pipeline Execution
    raw_df = generate_synthetic_telemetry_data(n_samples=sample_size)
    cleaned_df = clean_and_preprocess_data(raw_df)
    model, accuracy, ml_df = train_breakdown_classifier(cleaned_df)
    final_df = apply_ethical_guardrail(ml_df, threshold_m=aquifer_threshold)
    
    # Top Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_pumps = len(final_df)
    mechanic_cases = len(final_df[final_df["Action_Code"] == "DISPATCH_MECHANIC"])
    aquifer_cases = len(final_df[final_df["Action_Code"] == "OVERRIDE_AQUIFER"])
    normal_pumps = len(final_df[final_df["Action_Code"] == "NORMAL"])
    
    col1.metric("Total Pumping Units", total_pumps)
    col2.metric("Mechanical Risk (Action)", mechanic_cases, delta=f"{round(mechanic_cases/total_pumps*100, 1)}%", delta_color="inverse")
    col3.metric("Aquifer Override (Ethical)", aquifer_cases, delta="Drought Relabeled", delta_color="normal")
    col4.metric("Model Classification Acc.", f"{round(accuracy*100, 1)}%")
    
    st.markdown("---")
    
    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Capacity Dispatch & Field Plan", 
        "📊 Telemetry & ML Analytics", 
        "🛡️ Ethical Guardrail Verification", 
        "📁 Raw Data & Ingestion Hygiene"
    ])
    
    # TAB 1: CAPACITY DISPATCH
    with tab1:
        st.subheader("🚐 Junior Water Engineer Dispatch Action Plan")
        st.write(f"Prioritizing top mechanical risk cases capped by your monthly budget limit of **{max_mechanic_vans} dispatches**.")
        
        # Filter for Mechanic Dispatch
        dispatch_candidates = final_df[final_df["Action_Code"] == "DISPATCH_MECHANIC"].copy()
        
        # Compute Priority Composite Index: Risk Probability + Public Priority
        dispatch_candidates["Priority_Index"] = (
            dispatch_candidates["ML_Breakdown_Prob"] * 0.7 + 
            dispatch_candidates["Public_Utility_Priority"] * 0.3
        )
        
        dispatch_sorted = dispatch_candidates.sort_values(by="Priority_Index", ascending=False)
        
        # Allocated vs Overflow
        allocated_vans = dispatch_sorted.head(max_mechanic_vans)
        overflow_vans = dispatch_sorted.iloc[max_mechanic_vans:]
        
        st.markdown(f"**Allocated Mechanic Dispatches:** `{len(allocated_vans)}` / `{max_mechanic_vans}` Budgeted Capacity")
        
        # Display Table
        st.dataframe(
            allocated_vans[[
                "Pump_ID", "Gram_Panchayat", "Cluster", "Pump_Type", 
                "Recency_Days", "Energy_Discharge_Ratio", "OM_Reserve_INR", 
                "Public_Utility_Priority", "ML_Breakdown_Prob"
            ]].rename(columns={
                "Recency_Days": "Maintenance Recency (Days)",
                "Energy_Discharge_Ratio": "Energy Ratio (L/kWh)",
                "OM_Reserve_INR": "O&M Balance (₹)",
                "Public_Utility_Priority": "School/PHC Priority",
                "ML_Breakdown_Prob": "Breakdown Prob"
            }),
            use_container_width=True
        )
        
        # Download Dispatch CSV Button
        csv_buffer = io.StringIO()
        allocated_vans.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Field Dispatch Action Plan (CSV)",
            data=csv_buffer.getvalue(),
            file_name="JalDrishti_Mechanic_Dispatch_Plan.csv",
            mime="text/csv"
        )
        
        if len(overflow_vans) > 0:
            st.warning(f"⚠️ **Budget Overflow Warning:** {len(overflow_vans)} high-risk pumps could not be assigned a mechanic van due to budget constraints. Consider increasing dispatch capacity.")
            
    # TAB 2: ANALYTICS & VISUALIZATIONS
    with tab2:
        st.subheader("📊 Telemetry-RFM Feature Space & Clustering")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Scatter Plot: Energy Ratio vs Recency
            fig_scatter = px.scatter(
                final_df,
                x="Energy_Discharge_Ratio",
                y="Recency_Days",
                color="Final_Classification",
                size="Daily_Runtime_Hrs",
                hover_data=["Pump_ID", "Gram_Panchayat", "OM_Reserve_INR"],
                title="Mechanical Efficiency (L/kWh) vs Maintenance Recency (Days)",
                color_discrete_map={
                    "Optimal / Normal Operation": "#38A169",
                    "High Mechanical Risk (Mechanic Van Required)": "#E53E3E",
                    "Hydrogeological Stress (Tanker/Borewell Deepening)": "#DD6B20"
                }
            )
            fig_scatter.update_layout(legend=dict(orient="h", yanchor="bottom", y=-0.4, xanchor="left", x=0))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_b:
            # O&M Reserve Fund Distribution
            fig_box = px.box(
                final_df,
                x="Final_Classification",
                y="OM_Reserve_INR",
                color="Final_Classification",
                title="Gram Panchayat O&M Reserve Fund (₹) by Risk Category",
                color_discrete_map={
                    "Optimal / Normal Operation": "#38A169",
                    "High Mechanical Risk (Mechanic Van Required)": "#E53E3E",
                    "Hydrogeological Stress (Tanker/Borewell Deepening)": "#DD6B20"
                }
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    # TAB 3: ETHICAL GUARDRAIL
    with tab3:
        st.subheader("🛡️ Contextual Ethical Guardrail & Bias Prevention")
        st.markdown("""
        **Why this matters:** When groundwater tables collapse during extreme heatwaves, pumps must draw water from significantly greater depths, dropping the $L/kWh$ discharge ratio. 
        A naive AI model treats this drop as mechanical friction or operator negligence, triggering unfair technician penalties. 
        
        **JalDrishti's Guardrail** cross-references the Central Ground Water Board (CGWB) telemetry index:
        """)
        
        guardrail_triggered = final_df[final_df["Action_Code"] == "OVERRIDE_AQUIFER"]
        
        st.info(f"💡 **Guardrail Audit:** `{len(guardrail_triggered)}` pumps were successfully re-classified from **'Mechanical Failure'** to **'Hydrogeological Stress'** because regional groundwater dropped $> {aquifer_threshold}\text{ meters}$.")
        
        st.dataframe(
            guardrail_triggered[[
                "Pump_ID", "Gram_Panchayat", "Cluster", 
                "Energy_Discharge_Ratio", "CGWB_Aquifer_Drop_m", 
                "ML_Breakdown_Prob", "Final_Classification"
            ]].rename(columns={
                "Energy_Discharge_Ratio": "Low Discharge Ratio (L/kWh)",
                "CGWB_Aquifer_Drop_m": "Aquifer Drop Level (m)",
                "ML_Breakdown_Prob": "Original ML Risk Score"
            }),
            use_container_width=True
        )

    # TAB 4: RAW DATA & HYGIENE
    with tab4:
        st.subheader("📁 Synthetic Telemetry Raw Ingestion vs Preprocessed Data")
        st.markdown("Demonstrating ingestion hygiene converting messy strings (`₹ 12,500`, `110.2 L/kWh`) into standard floats.")
        
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("**Raw Messy Field Input Logs (Sample):**")
            st.dataframe(raw_df[["Pump_ID", "Recency_Raw", "OM_Reserve_Raw", "Energy_Discharge_Ratio_Raw"]].head(10))
            
        with col_d:
            st.markdown("**Cleaned & Parsed Telemetry Vectors:**")
            st.dataframe(cleaned_df[["Pump_ID", "Recency_Days", "OM_Reserve_INR", "Energy_Discharge_Ratio"]].head(10))

if __name__ == "__main__":
    main()
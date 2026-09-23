# JalDrishti: Rural Drinking Water Pump Failure Prediction

An ML-based prototype for predicting rural drinking-water pump breakdown risk and prioritizing limited maintenance resources.

## Project objective

JalDrishti demonstrates a complete data-to-decision pipeline:

1. Generate synthetic pump telemetry records.
2. Clean messy operational data.
3. Engineer maintenance and operational features.
4. Train a Random Forest classifier.
5. Apply a contextual groundwater-stress guardrail.
6. Prioritize mechanic dispatches under a monthly capacity limit.
7. Visualize results through a Streamlit dashboard.
8. Export a field-dispatch CSV.

## Features

- Recency of preventive maintenance
- Six-month tariff collection frequency
- O&M reserve balance
- Energy-to-discharge ratio
- Daily pumping runtime
- Public-utility priority
- Regional aquifer depth drop
- Random Forest breakdown-risk prediction
- Hydrogeological-stress override
- Capacity-constrained dispatch planning
- Plotly analytics
- CSV field-plan download

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit.

## Important limitation

This is a **synthetic-data coursework prototype**. The generated target labels are designed for demonstration of the ML pipeline and do not establish real-world predictive accuracy. A production system would require validated field data, domain review, monitoring, and independent model evaluation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

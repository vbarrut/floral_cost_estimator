# Floral Cost Estimation Tool

## Overview
The Floral Cost Estimation Tool is a web-based application built for **Faith Floral Studio**.  
It provides both customers and staff with a fast, transparent, and accurate way to generate wedding and event floral estimates.  
The tool runs entirely on **Streamlit Cloud**, meaning evaluators and clients do **not** need to install anything locally.

---

## Live Application
Access the deployed tool here:

👉 **https://floralcostestimator.streamlit.app/**

---

## Features

### **Customer Mode**
- Enter guest count and event season  
- Browse floral offerings organized by category  
- Select desired items and quantities  
- Automatic calculation of:
  - Subtotals  
  - Delivery fee  
  - Service fee (tiered)  
  - Tax  
  - Final total  
- Download an estimate as a CSV file  

### **Staff Mode** (password protected)
- Itemized materials, labor, and margin breakdowns  
- Stem-price-indexed materials scaling  
- Adjustable quantities  
- Internal tax, service fee, and delivery computations  
- Machine-learning price comparison  
- Visual analytics including:
  - Cost composition chart  
  - Feature importance diagram  
  - Predicted-vs-actual plot  

---

## Machine Learning Model
The system incorporates a **Random Forest Regressor** trained on more than **1,000 real floral line-item sales records**.  
The model provides a data-driven estimate of expected line-item totals, helping staff validate pricing consistency.

---

## Project Structure
app.py # Main Streamlit app
requirements.txt # Dependencies
README.md # Documentation
trained_model.pkl # Machine learning model
retrain_model.py # Script to retrain the model
visualizations/ # Feature importance, predicted vs actual
.streamlit/
├── secrets.toml # Staff password
└── config.toml # UI theme settings
data/
├──pricing_config.csv # Cleaned pricing configuration
├──sales_raw.csv # Historical sales dataset
└──flower_costs.csv # Per-stem flower pricing

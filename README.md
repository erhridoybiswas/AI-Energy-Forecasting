# AI-Powered Energy Consumption Forecasting System

## Overview
This project is a machine learning-based energy forecasting system that simulates realistic industrial energy consumption and predicts future usage using historical patterns, lag features, and time-based variables.

The project is designed as a beginner-friendly, industry-oriented proof-of-work portfolio project for GitHub, internships, and placements.

## Problem Statement
Organizations often face difficulties in estimating future energy usage. Poor forecasting can lead to:
- higher electricity bills
- inefficient resource planning
- peak demand penalties
- poor equipment scheduling
- energy waste

This project solves that by predicting future energy consumption from historical and simulated operational data.

## Industry Relevance
This project is useful for:
- smart cities
- electricity boards
- manufacturing plants
- data centers
- renewable energy companies
- commercial buildings

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Dataset
The project uses a realistic synthetic hourly dataset with:
- timestamp
- temperature
- occupancy index
- production index
- hour, day, month features
- energy consumption in kWh

## Architecture
1. Data generation
2. Preprocessing
3. Feature engineering
4. Model training
5. Evaluation
6. Forecasting
7. Visualization

## Installation

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Mac/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Usage
Run the full project with:
python main.py
Output Files
•	data/energy_data.csv
•	outputs/test_predictions.csv
•	outputs/forecast_output.csv
•	outputs/metrics.json
•	models/best_model.joblib
•	images/actual_vs_predicted.png
•	images/forecast_plot.png
•	images/feature_importance.png
Results
The project evaluates:
•	RMSE
•	MAE
•	R²
It also generates:
•	actual vs predicted graph
•	future forecast graph
•	feature importance plot
Screenshots
Add screenshots here after running the project:
•	Dataset preview
•	Training logs
•	Metrics output
•	Prediction graph
•	Forecast graph
•	GitHub repository preview
Learning Outcomes
•	time-series forecasting
•	feature engineering
•	model evaluation
•	data simulation
•	business-oriented analytics
•	GitHub portfolio building
Future Enhancements
•	LSTM forecasting
•	ARIMA/SARIMA comparison
•	Streamlit dashboard
•	real weather API integration
•	anomaly detection
•	electricity cost estimation
Author
**Hridoy Biswas**


import logging
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

# Load fraud dataset
fraud_data = pd.read_csv("fraud_data.csv")  # Ensure this file exists

# Configure logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/fraud-summary", methods=["GET"])
def fraud_summary():
    try:
        total_transactions = len(fraud_data)
        total_fraud_cases = fraud_data["is_fraud"].sum()
        fraud_percentage = (total_fraud_cases / total_transactions) * 100

        summary = {
            "total_transactions": total_transactions,
            "total_fraud_cases": int(total_fraud_cases),
            "fraud_percentage": round(fraud_percentage, 2)
        }

        logging.info("Fraud summary requested.")
        return jsonify(summary)

    except Exception as e:
        logging.error(f"Error in /fraud-summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

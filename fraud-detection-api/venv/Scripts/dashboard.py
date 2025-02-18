import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import requests

# Load fraud data
fraud_data = pd.read_csv("fraud_data.csv")  # Ensure this file exists

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

    # Summary Boxes
    html.Div(id="summary-boxes", style={"display": "flex", "justifyContent": "space-around"}),

    # Fraud Trends Over Time
    dcc.Graph(id="fraud-trend"),

    # Fraud Distribution by Location
    dcc.Graph(id="fraud-location"),

    # Fraud Cases by Device and Browser
    dcc.Graph(id="fraud-device-browser"),
])

@app.callback(
    Output("summary-boxes", "children"),
    Input("fraud-trend", "id")  # Trigger update
)
def update_summary(_):
    """Fetch fraud summary from Flask API and update summary boxes."""
    response = requests.get("http://127.0.0.1:5000/fraud-summary")
    data = response.json()

    return [
        html.Div(f"Total Transactions: {data['total_transactions']}", style={"fontSize": "20px"}),
        html.Div(f"Total Fraud Cases: {data['total_fraud_cases']}", style={"fontSize": "20px"}),
        html.Div(f"Fraud Percentage: {data['fraud_percentage']}%", style={"fontSize": "20px"}),
    ]

@app.callback(
    Output("fraud-trend", "figure"),
    Input("summary-boxes", "children")  # Trigger update
)
def update_fraud_trend(_):
    """Generate a line chart of fraud cases over time."""
    fraud_over_time = fraud_data.groupby("transaction_date")["is_fraud"].sum().reset_index()
    fig = px.line(fraud_over_time, x="transaction_date", y="is_fraud", title="Fraud Cases Over Time")
    return fig

@app.callback(
    Output("fraud-location", "figure"),
    Input("summary-boxes", "children")
)
def update_fraud_location(_):
    """Generate a bar chart showing fraud cases by location."""
    fraud_by_location = fraud_data.groupby("location")["is_fraud"].sum().reset_index()
    fig = px.bar(fraud_by_location, x="location", y="is_fraud", title="Fraud Cases by Location")
    return fig

@app.callback(
    Output("fraud-device-browser", "figure"),
    Input("summary-boxes", "children")
)
def update_device_browser(_):
    """Generate a bar chart comparing fraud cases by device and browser."""
    fraud_by_device = fraud_data.groupby("device_type")["is_fraud"].sum().reset_index()
    fraud_by_browser = fraud_data.groupby("browser")["is_fraud"].sum().reset_index()

    fig = px.bar(fraud_by_device, x="device_type", y="is_fraud", title="Fraud Cases by Device")
    fig2 = px.bar(fraud_by_browser, x="browser", y="is_fraud", title="Fraud Cases by Browser")

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

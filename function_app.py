import azure.functions as func
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from azure.storage.blob import BlobServiceClient
from azure.ai.openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
import os
import io

app = func.FunctionApp()

# Get data
def get_historical_data():
    """Load historical data from Azure Blob Storage."""
    connection_string = os.environ["AzureWebJobsStorage"]
    container_name = "drug-orders"
    blob_name = "historical_orders.csv"
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        blob_data = blob_client.download_blob()
        df = pd.read_csv(io.StringIO(blob_data.content_as_text()))
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    except Exception as e:
        logging.error(f"Error loading historical data: {str(e)}")
        raise

def calculate_statistics(historical_data: pd.DataFrame, current_order: dict) -> dict:
    """Calculate statistical metrics for the specific drug and ward."""
    
    # Filter for the specific drug and ward
    mask = (historical_data['Ward'] == current_order['Ward']) & \
           (historical_data['Item'] == current_order['Item'])
    relevant_history = historical_data[mask].copy()
    
    if len(relevant_history) == 0:
        return {
            "error": "No historical data found for this drug/ward combination"
        }

    # Basic statistics
    stats = {
        "avg_quantity": float(relevant_history['Quantity'].mean()),
        "median_quantity": float(relevant_history['Quantity'].median()),
        "std_dev": float(relevant_history['Quantity'].std()),
        "min_quantity": float(relevant_history['Quantity'].min()),
        "max_quantity": float(relevant_history['Quantity'].max()),
        "total_orders": len(relevant_history),
        "units": current_order['Units']
    }
    
    # Calculate Z-score for current order
    if stats["std_dev"] > 0:
        stats["z_score"] = float((current_order['Quantity'] - stats["avg_quantity"]) / stats["std_dev"])
    else:
        stats["z_score"] = 0.0

    # Recent trend analysis (last 30 days)
    thirty_days_ago = pd.to_datetime(current_order['Date']) - timedelta(days=30)
    recent_data = relevant_history[relevant_history['Date'] >= thirty_days_ago]
    
    if len(recent_data) >= 2:
        # Calculate simple linear regression for trend
        x = (recent_data['Date'] - recent_data['Date'].min()).dt.days
        y = recent_data['Quantity']
        coefficients = np.polyfit(x, y, 1)
        stats["trend_slope"] = float(coefficients[0])  # Change per day
        stats["trend_direction"] = "increasing" if coefficients[0] > 0 else "decreasing" if coefficients[0] < 0 else "stable"
    else:
        stats["trend_slope"] = 0.0
        stats["trend_direction"] = "insufficient_data"

    return stats

def generate_ai_insights(stats: dict, current_order: dict) -> str:
    """Generate insights using Azure OpenAI with managed identity."""
    try:
        # Use managed identity credentials
        credential = DefaultAzureCredential()
        
        # Initialize OpenAI client with managed identity
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            credential=credential,
            api_version="2024-02-15-preview"
        )
        
        prompt = f"""
        As a pharmaceutical analytics assistant, analyze this drug order:

        Drug: {current_order['Item']}
        Ward: {current_order['Ward']}
        Current order quantity: {current_order['Quantity']} {current_order['Units']}
        Date: {current_order['Date']}

        Statistical Context:
        - Historical average: {stats['avg_quantity']:.1f} {current_order['Units']}
        - Standard deviation: {stats['std_dev']:.1f}
        - Z-score of current order: {stats['z_score']:.1f}
        - Recent trend: {stats['trend_direction']} ({stats['trend_slope']:.3f} units/day)
        - Total historical orders analyzed: {stats['total_orders']}
        - Historical range: {stats['min_quantity']} to {stats['max_quantity']} {current_order['Units']}

        Provide a concise analysis (2-3 sentences) addressing:
        1. How this order compares to historical patterns
        2. Any notable variations or patterns
        3. Specific considerations for the pharmacist
        """

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a pharmaceutical analytics assistant providing brief, clinical insights about drug orders based on historical patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating AI insights: {str(e)}")
        return "Error generating AI insights. Please rely on the statistical analysis."

@app.route(route="analyze_order", auth_level=func.AuthLevel.FUNCTION)
async def analyze_order(req: func.HttpRequest) -> func.HttpResponse:
    """Main Azure Function handler."""
    try:
        # Parse request data
        req_body = req.get_json()
        current_order = {
            "Date": req_body.get("orderDate"),
            "Ward": int(req_body.get("ward")),
            "Item": req_body.get("item"),
            "Units": req_body.get("units"),
            "Quantity": float(req_body.get("quantity"))
        }
        
        # Load historical data
        historical_data = get_historical_data()
        
        # Calculate statistics
        stats = calculate_statistics(historical_data, current_order)
        
        if "error" in stats:
            return func.HttpResponse(
                json.dumps({"error": stats["error"]}),
                mimetype="application/json",
                status_code=404
            )
        
        # Generate AI insights
        insights = generate_ai_insights(stats, current_order)
        
        # Prepare response
        response = {
            "orderDetails": current_order,
            "statistics": stats,
            "aiInsights": insights,
            "analysisTimestamp": datetime.utcnow().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(response, default=str),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
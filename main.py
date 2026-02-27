from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap

# Initialize FastAPI app
app = FastAPI()

# Load trained model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_credit_model.pkl")

model = joblib.load(MODEL_PATH)


# Business threshold (profit-optimized)
THRESHOLD = 0.1189


# Input schema
class CreditInput(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def home():
    return {"message": "Credit Risk API is running 🚀"}


@app.post("/predict")
def predict(data: CreditInput):

    # Convert input to dataframe
    input_df = pd.DataFrame([data.dict()])

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # Business decision
    decision = "REJECT" if probability > THRESHOLD else "APPROVE"

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)


    # Convert SHAP output safely to Python float
    feature_impact = {
        feature: float(value)
        for feature, value in zip(input_df.columns, shap_values[0])
    }

    # Top 5 risk drivers
    top_features = sorted(
        feature_impact.items(),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[:5]

    explanation = {
        feature: float(value)
        for feature, value in top_features
    }

    return {
        "default_probability": float(probability),
        "decision": decision,
        "top_risk_drivers": explanation
    }

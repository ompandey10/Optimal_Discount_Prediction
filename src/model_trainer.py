import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def train_and_evaluate():
    df = pd.read_csv(os.path.join(DATA_DIR, 'retail_data.csv'))
    
    X = df.drop('Expected_Sales', axis=1)
    y = df['Expected_Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['Original_Price', 'Proposed_Discount', 'Historical_Sales', 
                       'Customer_Demand', 'Promotion_Status', 'Competitor_Price']
    categorical_features = ['Product_Category', 'Season']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
        
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        
        print(f"--- {name} ---")
        print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = name
            
    print(f"Best Model: {best_model_name} with R2: {best_r2:.4f}")
    
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    train_and_evaluate()

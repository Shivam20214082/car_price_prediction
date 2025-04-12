# l_baseline_xgboost.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import os

def baseline_xgboost(df):
    # Define target and features
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Mapping categorical variables to the mean price for each category
    for col in ['company', 'fuel_type', 'model']:
        means = df.groupby(col)['Price'].mean()
        X[col] = X[col].map(means)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost Regressor with the best parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        subsample=0.8,
        n_estimators=200, 
        max_depth=10,
        learning_rate=0.05,
        gamma=5,
        colsample_bytree=0.6,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot Actual vs Predicted Prices
    df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    fig = px.scatter(df_pred, x='Actual', y='Predicted',
                     title='XgBoost - Actual vs Predicted Car Prices',
                     labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
                     trendline='ols')
    fig.update_traces(marker=dict(size=8, color='dodgerblue'), selector=dict(mode='markers'))

    # Save figure to the 'output' folder
    output_folder = 'outputs'  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    fig.write_image(os.path.join(output_folder, 'XgBoost_actual_vs_predicted_prices.png'))  # Save as PNG

    # Return evaluation metrics for the combined model
    return mae, rmse, r2

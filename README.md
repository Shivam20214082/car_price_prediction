# 🚗 Car Price Prediction

This project predicts car prices using a combination of machine learning and deep learning models including:

- ✅ XGBoost (baseline)
- 🔁 RT-Transformer
- 🔁 Crossformer
- 🔁 TabTransformer

The goal is to evaluate model performances and identify the best approach for real-world car price forecasting.

## 📁 Project Structure

car_price_prediction/
├── combined_model.py             # Main script to run all models  
├── data/  
│   └── car_data.csv              # Input dataset  
├── scripts/  
│   ├── baseline_xgboost.py       # XGBoost model  
│   ├── crossformer.py            # Crossformer model  
│   ├── rt_transformer.py         # RT-Transformer model  
│   └── tabtransformer.py         # TabTransformer model  
├── notebooks/  
│   ├── 1_baseline_xgboost.ipynb          
|   ├── 2_rt_transformer.ipynb
│   ├── 3_crossformer.ipynb
│   └── 5_tabtransformer.ipynb
├── outputs/  
│   └── model_result.csv           # Final prediction outputs  
└── README.md                     # Project documentation



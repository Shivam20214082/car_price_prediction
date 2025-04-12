import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.baseline_xgboost import baseline_xgboost
from scripts.rt_transformer import rt_transformer
from scripts.crossformer import crossformer
from scripts.tabtransformer import tabtransformer

def print_header(msg):
    print("\n" + "=" * 60)
    print(f"{msg}")
    print("=" * 60)

def main():
    print_header("🚗 Car Price Prediction - Model Evaluation Started")

    print("📥 Loading datasets...")
    df = pd.read_csv('data/car_data.csv')
    df4 = pd.read_csv('data/car_data4.csv')

    print("🧹 Cleaning data...")
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df.drop_duplicates()
    df = pd.concat([df4, df], ignore_index=True)
    df['model'] = df['name'].apply(lambda x: ' '.join(x.split()[1:]))
    df['car_age'] = 2025 - df['year']
    df.drop(['name', 'year'], axis=1, inplace=True)

    results = {}

    print_header("🚀 Running Baseline XGBoost Model")
    mae, rmse, r2 = baseline_xgboost(df)
    results['baseline_xgboost'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print_header("🤖 Running RT-Transformer Model")
    mae, rmse, r2 = rt_transformer(df)
    results['rt_transformer'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print_header("🔁 Running CrossFormer Model")
    mae, rmse, r2 = crossformer(df)
    results['crossformer'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print_header("📊 Running TabTransformer Model")
    mae, rmse, r2 = tabtransformer(df)
    results['tabtransformer'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print_header("📈 Model Results Summary")
    results_df = pd.DataFrame(results).T
    print(results_df)

    results_df.to_csv('outputs/model_results.csv', index=True)
    print("\n✅ Results saved to 'outputs/model_results.csv'")
    print("🏁 All models evaluated successfully!\n")

if __name__ == "__main__":
    main()

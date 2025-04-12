import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import os
import plotly.express as px


# Preprocessing and data scaling function
def preprocess_data(df):
    for col in ['company', 'fuel_type', 'model']:
        df[col] = df[col].astype('category')
        df[col + '_cat'] = df[col].cat.codes

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    features = df[['car_age', 'kms_driven', 'company_cat', 'fuel_type_cat', 'model_cat']].values
    target = df['Price'].values.reshape(-1, 1)

    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

    # Create sequences
    seq_len, pred_len = 36, 12
    X, y = [], []
    for i in range(len(features_scaled) - seq_len - pred_len):
        X.append(features_scaled[i:i + seq_len])
        y.append(target_scaled[i + seq_len:i + seq_len + pred_len])

    X = np.array(X)
    y = np.array(y)

    # Return the scaled features and target
    return X, y, scaler_y

# Crossformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CrossformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        return self.norm2(x + x2)

class Crossformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, d_ff=256, num_layers=3, pred_len=12, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(*[CrossformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, pred_len)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global pooling
        return self.regressor(x)

# Dataset Class
class CarPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Crossformer Training and Evaluation Function
def crossformer(df):
    # Preprocess data
    X, y, scaler_y = preprocess_data(df)

    # Split data into train and test
    split = int(0.8 * len(X))
    train_ds = CarPriceDataset(X[:split], y[:split])
    test_ds = CarPriceDataset(X[split:], y[split:])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    # Initialize model, loss function, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Crossformer(input_dim=X.shape[2]).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Ensure models/ directory exists
    os.makedirs('models', exist_ok=True)

    epochs = 50
    best_loss = float('inf')
    patience, wait = 7, 0
    model_path = 'models/crossformer_best_model.pt'

    for ep in range(epochs):
        model.train()
        loss_sum = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        scheduler.step()
        val_loss /= len(test_dl)
        print(f"Epoch {ep+1}, Train Loss: {loss_sum/len(train_dl):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait > patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate model
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_true.append(yb.numpy())
            y_pred.append(pred)

    y_true = scaler_y.inverse_transform(np.concatenate(y_true))
    y_pred = scaler_y.inverse_transform(np.concatenate(y_pred))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Plot Actual vs Predicted Prices
    df_pred = pd.DataFrame({
        'Actual': np.concatenate(y_true).flatten(),
        'Predicted': np.concatenate(y_pred).flatten()
    })

    fig = px.scatter(df_pred, x='Actual', y='Predicted',
                     title='crossformer - Actual vs Predicted Car Prices',
                     labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
                     trendline='ols')
    fig.update_traces(marker=dict(size=8, color='dodgerblue'), selector=dict(mode='markers'))

    # Ensure outputs/ directory exists
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # Save plot
    fig.write_image(os.path.join(output_folder, 'crossformer_actual_vs_predicted_prices.png'))

    return mae, rmse, r2



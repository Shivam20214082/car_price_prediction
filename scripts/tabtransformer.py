# tabtransformer.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# === TabTransformer Model Definition ===
class TabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, dim=32, depth=2, heads=4, mlp_hidden=64, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(num_categories, dim) for num_categories in categories
        ])
        self.embedding_dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, len(categories), dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True),
            num_layers=depth
        )
        total_features = dim * len(categories) + num_continuous
        self.norm = nn.LayerNorm(total_features)
        self.mlp = nn.Sequential(
            nn.Linear(total_features, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(mlp_hidden + dim * len(categories) + num_continuous, 1)

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
        x_cat = torch.stack(embeds, dim=1)              # (B, C, D)
        x_cat = self.transformer(x_cat)                 # (B, C, D)
        x_cat = x_cat.flatten(1)                        # (B, C*D)
        x = torch.cat([x_cat, x_num], dim=1)            # (B, C*D + num_continuous)
        x = self.norm(x)
        x_mlp = self.mlp(x)                             # (B, mlp_hidden)
        out = self.head(torch.cat([x_mlp, x], dim=1))   # Concatenate for residual-style use
        return out.squeeze()

# === Data Preprocessing and Training Loop ===
def tabtransformer(df):
    # Data Preprocessing
    df['Price'] = np.log1p(df['Price'])

    for col in ['company', 'fuel_type', 'model']:
        df[col] = df[col].astype('category')
        df[col + '_cat'] = df[col].cat.codes

    cat_cols = ['company_cat', 'fuel_type_cat', 'model_cat']
    num_cols = ['car_age', 'kms_driven']
    target_col = 'Price'

    X_categorical = torch.tensor(df[cat_cols].values, dtype=torch.long)
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[num_cols])
    X_numerical = torch.tensor(X_numerical, dtype=torch.float32)
    y_tensor = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    # Train-Test Split
    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_categorical, X_numerical, y_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(X_cat_train, X_num_train, y_train)
    val_dataset = TensorDataset(X_cat_val, X_num_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # === Model Initialization ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories = [df[col].nunique() for col in cat_cols]
    num_cont = len(num_cols)

    model = TabTransformer(categories, num_cont, dim=32, depth=2, heads=4, mlp_hidden=128, dropout=0.2).to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # === Train Loop ===
    best_r2 = -float('inf')
    patience = 10
    counter = 0
    best_model_path = '../best_model.pt'

    epochs = 100
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x_cat, x_num)
            loss = loss_fn(preds, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                x_cat, x_num = x_cat.to(device), x_num.to(device)
                pred = model(x_cat, x_num).cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(y.numpy())

        mae = mean_absolute_error(val_targets, val_preds)
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        r2 = r2_score(val_targets, val_preds)
        val_losses.append(mae)
        scheduler.step(mae)

        print(f"Epoch {epoch+1:03}/{epochs} | Train Loss: {avg_train_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print("\u23f9\ufe0f Early stopping triggered!")
                break

    # === Final Evaluation ===
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    final_preds, final_targets = [], []
    with torch.no_grad():
        for x_cat, x_num, y in val_loader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            pred = model(x_cat, x_num).cpu().numpy()
            final_preds.extend(pred)
            final_targets.extend(y.numpy())

    final_preds = np.expm1(np.array(final_preds).flatten())
    final_targets = np.expm1(np.array(final_targets).flatten())

    mae = mean_absolute_error(final_targets, final_preds)
    rmse = np.sqrt(mean_squared_error(final_targets, final_preds))
    r2 = r2_score(final_targets, final_preds)

    print(f"\n✅ Final Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # === Save Results ===
    output_folder = 'outputs/'
    os.makedirs(output_folder, exist_ok=True)

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(8, 6))
    plt.scatter(final_targets, final_preds, alpha=0.7, edgecolors='k')
    plt.plot([min(final_targets), max(final_targets)], [min(final_targets), max(final_targets)], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices (TabTransformer)')
    plt.savefig(os.path.join(output_folder, 'TabTransformer_actual_vs_predicted_tabtransformer.png'))
    plt.close()

    return mae, rmse, r2

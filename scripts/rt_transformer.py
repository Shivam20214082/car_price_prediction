import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
import os

# Function to train and evaluate the RT Transformer model
def rt_transformer(df):
    # Convert to category and store category codes
    for col in ['company', 'fuel_type', 'model']:
        df[col] = df[col].astype('category')
        df[col + '_cat'] = df[col].cat.codes
    # Categorical columns for embedding
    cat_cols = ['company_cat', 'fuel_type_cat', 'model_cat']
    cat_dims = [df[col].nunique() for col in cat_cols]  # e.g., [14, 3, 187]
    emb_dims = [(n, min(50, (n + 1) // 2)) for n in cat_dims]  # embedding size rule

    # Numerical columns
    num_cols = ['car_age', 'kms_driven']

    # Target
    target = 'Price'

    # Separate features
    X_cat = df[cat_cols].values
    X_num = df[num_cols].values
    y = df[target].values

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Train-test split
    X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_cat, X_num_scaled, y, test_size=0.2, random_state=42
    )

    X_num_train = scaler.fit_transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    class CarDataset(Dataset):
        def __init__(self, X_cat, X_num, y):
            self.X_cat = torch.tensor(X_cat, dtype=torch.long)  # for embedding
            self.X_num = torch.tensor(X_num, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X_cat)

        def __getitem__(self, idx):
            return self.X_cat[idx], self.X_num[idx], self.y[idx]

    # Datasets
    train_ds = CarDataset(X_cat_train, X_num_train, y_train)
    test_ds = CarDataset(X_cat_test, X_num_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    class CarPriceModel(nn.Module):
        def __init__(self, cat_dims, emb_dims, num_input_dim):
            super().__init__()
            self.emb_layers = nn.ModuleList([
                nn.Embedding(input_dim, emb_dim) 
                for input_dim, emb_dim in zip(cat_dims, emb_dims)
            ])
            total_emb_dim = sum(emb_dims)
            self.fc = nn.Sequential(
                nn.Linear(total_emb_dim + num_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x_cat, x_num):
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
            x = torch.cat(embeddings + [x_num], dim=1)
            return self.fc(x)

    # Initialize model, criterion, and optimizer
    cat_dims = [len(df[col].unique()) for col in cat_cols]
    emb_dims = [(dim, min(50, (dim + 1) // 2)) for dim in cat_dims]  # embedding size rule
    num_input_dim = X_num_train.shape[1]

    model = CarPriceModel(cat_dims, [e[1] for e in emb_dims], num_input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for cat_x, num_x, y in train_dl:
            optimizer.zero_grad()
            preds = model(cat_x, num_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1} Loss: {epoch_loss/len(train_dl):.2f}')

    # Evaluate the model
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for cat_x, num_x, y in test_dl:
            preds = model(cat_x, num_x)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())

    # Combine all batches
    y_pred = np.vstack(all_preds).flatten()
    y_true = np.vstack(all_targets).flatten()

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R² Score: {r2:.4f}')

    # Save the plot to the output folder instead of showing it
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted vs Actual'))
    fig.add_trace(go.Scatter(x=y_true, y=y_true, mode='lines', name='Ideal Line'))
    fig.update_layout(title='RT_Transformer - Actual vs Predicted Prices',
                      xaxis_title='Actual Price',
                      yaxis_title='Predicted Price',
                      height=500)
    
    # Save the plot as a PNG image
    fig.write_image(os.path.join(output_folder, "RT_Transformer_actual_vs_predicted_prices.png"))

    # ✅ Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/rt_transformer_model.pth')

    # Return metrics for integration with the combined_model.py
    return mae, rmse, r2

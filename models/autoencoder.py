import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import SENSOR_COLUMNS


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class AutoencoderDetector:
    def __init__(self, threshold_percentile=95):
        self.scaler = StandardScaler()
        self.model = Autoencoder(input_dim=len(SENSOR_COLUMNS))
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def fit(self, df, epochs=30, lr=1e-3):
        x = df[SENSOR_COLUMNS]
        x_scaled = self.scaler.fit_transform(x)

        tensor_x = torch.tensor(x_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(tensor_x)
            loss = loss_fn(reconstructed, tensor_x)
            loss.backward()
            optimizer.step()

        errors = self._reconstruction_errors(df)
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def _reconstruction_errors(self, df):
        x = df[SENSOR_COLUMNS]
        x_scaled = self.scaler.transform(x)

        tensor_x = torch.tensor(x_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_x)

        errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1)

        return errors.numpy()

    def predict(self, df):
        errors = self._reconstruction_errors(df)

        return (errors > self.threshold).astype(int)

    def anomaly_score(self, df):
        return self._reconstruction_errors(df)
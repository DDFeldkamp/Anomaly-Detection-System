from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import SENSOR_COLUMNS


class IsolationForestDetector:
    def __init__(self, contamination=0.08):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=150
        )

    def fit(self, df):
        x = df[SENSOR_COLUMNS]
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled)

    def predict(self, df):
        x = df[SENSOR_COLUMNS]
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled)

        return (preds == -1).astype(int)

    def anomaly_score(self, df):
        x = df[SENSOR_COLUMNS]
        x_scaled = self.scaler.transform(x)

        return -self.model.decision_function(x_scaled)
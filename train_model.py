"""
SmartAir ML Model Trainer
=========================
• First run: generates synthetic dataset, trains WLS regression + RandomForest, saves .pkl
• Subsequent runs: loads real-time data from data/realtime_log.csv, retrains, overwrites .pkl
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

MODEL_PATH   = "models/aqi_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
DATA_PATH    = "data/realtime_log.csv"
SYNTH_PATH   = "data/synthetic_dataset.csv"
META_PATH    = "models/model_meta.json"

FEATURES = ["mq2", "mq135", "temperature", "humidity",
            "mq2_roll3", "mq135_roll3", "aqi_lag1", "aqi_lag2", "hour_sin", "hour_cos"]


def calc_aqi(mq2, mq135):
    raw = (mq2 * 0.4 + mq135 * 0.6) / 4095 * 500
    return float(np.clip(raw, 0, 500))


def aqi_label(aqi):
    if aqi < 51:  return "Good"
    if aqi < 101: return "Moderate"
    if aqi < 151: return "Sensitive"
    if aqi < 201: return "Unhealthy"
    if aqi < 301: return "Very Unhealthy"
    return "Hazardous"


def generate_synthetic_dataset(n=5000):
    """Generate realistic synthetic sensor data with temporal patterns."""
    print("📊 Generating synthetic dataset...")
    np.random.seed(42)
    
    hours = np.linspace(0, 24 * (n // 24), n)
    hour_of_day = hours % 24

    # Realistic diurnal patterns
    base_mq2   = 800 + 400 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/3) + np.random.normal(0, 80, n)
    base_mq135 = 1200 + 600 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/4) + np.random.normal(0, 120, n)
    temp       = 22 + 8 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2) + np.random.normal(0, 1.5, n)
    humidity   = 55 + 20 * np.sin(2 * np.pi * hour_of_day / 24 + np.pi/3) + np.random.normal(0, 5, n)

    # Random pollution spikes
    spike_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
    base_mq2[spike_idx]   *= np.random.uniform(1.5, 3.0, len(spike_idx))
    base_mq135[spike_idx] *= np.random.uniform(1.5, 2.5, len(spike_idx))

    base_mq2   = np.clip(base_mq2,   0, 4095)
    base_mq135 = np.clip(base_mq135, 0, 4095)
    temp       = np.clip(temp,       -10, 50)
    humidity   = np.clip(humidity,   0, 100)

    aqi = np.array([calc_aqi(m2, m135) for m2, m135 in zip(base_mq2, base_mq135)])

    # Future AQI (30-min ahead target) with trend
    aqi_future = np.roll(aqi, -int(30 * 60 / 4))  # 30min / 4sec poll
    aqi_future[-450:] = aqi[-450:]  # fill end

    df = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()] * n,
        "mq2": base_mq2.astype(int),
        "mq135": base_mq135.astype(int),
        "temperature": np.round(temp, 1),
        "humidity": np.round(humidity, 1),
        "aqi": np.round(aqi, 1),
        "aqi_future_30m": np.round(aqi_future, 1),
        "hour": hour_of_day
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv(SYNTH_PATH, index=False)
    print(f"✅ Synthetic dataset saved: {len(df)} rows → {SYNTH_PATH}")
    return df


def engineer_features(df):
    """Add rolling and lag features."""
    df = df.copy().reset_index(drop=True)
    df["mq2_roll3"]    = df["mq2"].rolling(3, min_periods=1).mean()
    df["mq135_roll3"]  = df["mq135"].rolling(3, min_periods=1).mean()
    df["aqi_lag1"]     = df["aqi"].shift(1).fillna(df["aqi"])
    df["aqi_lag2"]     = df["aqi"].shift(2).fillna(df["aqi"])
    hour = df.get("hour", pd.Series(np.zeros(len(df))))
    df["hour_sin"]     = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * hour / 24)
    return df


def train(df, source_name="synthetic"):
    """Train ensemble model and save pkl."""
    print(f"\n🤖 Training on {source_name} data ({len(df)} rows)...")

    df = engineer_features(df)
    df.dropna(subset=FEATURES + ["aqi_future_30m"], inplace=True)

    X = df[FEATURES].values
    y = df["aqi_future_30m"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Ensemble: GBR + RF + Ridge blend
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                    max_depth=4, subsample=0.8, random_state=42)
    rf  = RandomForestRegressor(n_estimators=150, max_depth=8,
                                min_samples_leaf=3, random_state=42)
    ridge = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("scaler", StandardScaler()),
                      ("reg",   Ridge(alpha=10.0))])

    gbr.fit(X_tr_s, y_train)
    rf.fit(X_tr_s, y_train)
    ridge.fit(X_tr_s, y_train)

    # Blend predictions
    def predict_ensemble(X_scaled):
        p_gbr   = gbr.predict(X_scaled)
        p_rf    = rf.predict(X_scaled)
        p_ridge = ridge.predict(X_scaled)
        return np.clip(0.45 * p_gbr + 0.40 * p_rf + 0.15 * p_ridge, 0, 500)

    y_pred = predict_ensemble(X_te_s)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"   MAE: {mae:.2f}  |  R²: {r2:.4f}")

    bundle = {
        "gbr": gbr, "rf": rf, "ridge": ridge,
        "scaler": scaler, "features": FEATURES
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "source": source_name,
        "rows": len(df),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
        "features": FEATURES
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata saved → {META_PATH}")
    return meta


def load_and_predict(mq2, mq135, temp, hum, history_aqi):
    """Load model, predict next 30-min AQI."""
    if not os.path.exists(MODEL_PATH):
        return calc_aqi(mq2, mq135)

    bundle  = joblib.load(MODEL_PATH)
    scaler  = bundle["scaler"]
    gbr     = bundle["gbr"]
    rf      = bundle["rf"]
    ridge   = bundle["ridge"]

    h = list(history_aqi) if history_aqi else [calc_aqi(mq2, mq135)]
    aqi_now = calc_aqi(mq2, mq135)
    lag1 = h[-1] if len(h) >= 1 else aqi_now
    lag2 = h[-2] if len(h) >= 2 else lag1
    roll3 = float(np.mean(h[-3:])) if len(h) >= 1 else aqi_now

    hour = datetime.now().hour + datetime.now().minute / 60
    feat = np.array([[mq2, mq135, temp, hum,
                      roll3, roll3, lag1, lag2,
                      np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)]])

    X_s = scaler.transform(feat)
    p_gbr   = gbr.predict(X_s)[0]
    p_rf    = rf.predict(X_s)[0]
    p_ridge = ridge.predict(X_s)[0]
    pred = float(np.clip(0.45*p_gbr + 0.40*p_rf + 0.15*p_ridge, 0, 500))
    return round(pred, 1)


def run():
    """Main entry point: first run = synthetic, subsequent = real data."""
    real_exists = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 200

    if real_exists:
        print(f"\n🔄 Real-time data found at {DATA_PATH} — retraining on actual data...")
        df = pd.read_csv(DATA_PATH)
        if "aqi_future_30m" not in df.columns:
            df["aqi"] = df.apply(lambda r: calc_aqi(r["mq2"], r["mq135"]), axis=1)
            df["aqi_future_30m"] = df["aqi"].shift(-450).fillna(df["aqi"])
        if "hour" not in df.columns:
            try:
                df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            except:
                df["hour"] = 0
        if len(df) < 50:
            print("⚠️  Too few real rows — augmenting with synthetic...")
            synth = generate_synthetic_dataset(2000)
            df = pd.concat([synth, df], ignore_index=True)
        meta = train(df, source_name="real+synthetic" if len(df) < 500 else "real")
    else:
        print("\n🚀 First run — using synthetic dataset...")
        synth = generate_synthetic_dataset()
        meta  = train(synth, source_name="synthetic")

    print(f"\n🎉 Training complete! Model performance:")
    print(f"   Source : {meta['source']}")
    print(f"   Rows   : {meta['rows']}")
    print(f"   MAE    : {meta['mae']}")
    print(f"   R²     : {meta['r2']}")


if __name__ == "__main__":
    run()

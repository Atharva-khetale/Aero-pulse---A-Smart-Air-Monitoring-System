"""
SmartAir Flask Server
=====================
• Polls ESP32 at http://192.168.4.1/data every 4 seconds
• Serves real-time dashboard, predict, and advice screens
• Logs data to CSV, retrains ML model on each run
• REST API for frontend JS polling
"""

import os, json, csv, threading, time, hashlib
from datetime import datetime
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_model import load_and_predict, calc_aqi, aqi_label, run as train_run

try:
    import urllib.request as urlreq
    NET_OK = True
except ImportError:
    NET_OK = False

# ── Config ──────────────────────────────────────────────────────────────────
ESP_URL      = "http://192.168.4.1"
POLL_SECS    = 4
DATA_PATH    = "data/realtime_log.csv"
USER_FILE    = "data/users.json"
SESSION_FILE = "data/session.json"
SECRET_KEY   = "smartair_secret_2025"

app = Flask(__name__)
app.secret_key = SECRET_KEY

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Data Store ───────────────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self.mq2         = 0
        self.mq135       = 0
        self.temperature = 0.0
        self.humidity    = 0.0
        self.aqi         = 0.0
        self.aqi_history = []
        self.connected   = False
        self.last_ts     = None
        self._lock       = threading.Lock()
        # Demo mode: simulate data if no ESP
        self._demo_t     = 0

    def update(self, raw):
        with self._lock:
            self.mq2         = raw.get("mq2", 0)
            self.mq135       = raw.get("mq135", 0)
            self.temperature = raw.get("temperature", 0.0)
            self.humidity    = raw.get("humidity", 0.0)
            self.aqi         = calc_aqi(self.mq2, self.mq135)
            self.aqi_history.append(self.aqi)
            if len(self.aqi_history) > 500:
                self.aqi_history.pop(0)
            self.connected = True
            self.last_ts   = datetime.now().isoformat()
            self._log()

    def demo_tick(self):
        """Generate realistic demo data when ESP not connected."""
        import math, random
        self._demo_t += 1
        t = self._demo_t
        mq2   = int(800 + 400 * math.sin(t * 0.05) + random.gauss(0, 30))
        mq135 = int(1200 + 500 * math.sin(t * 0.04 + 1) + random.gauss(0, 50))
        temp  = round(24 + 4 * math.sin(t * 0.02) + random.gauss(0, 0.3), 1)
        hum   = round(60 + 15 * math.sin(t * 0.03 + 2) + random.gauss(0, 2), 1)
        self.update({"mq2": max(0, min(4095, mq2)),
                     "mq135": max(0, min(4095, mq135)),
                     "temperature": max(-10, min(50, temp)),
                     "humidity": max(0, min(100, hum))})

    def _log(self):
        is_new = not os.path.exists(DATA_PATH)
        with open(DATA_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp","mq2","mq135","temperature","humidity","aqi"])
            w.writerow([self.last_ts, self.mq2, self.mq135,
                        self.temperature, self.humidity, round(self.aqi, 1)])

    def snapshot(self):
        with self._lock:
            lbl, col = aqi_label(self.aqi)
            return {
                "mq2": self.mq2,
                "mq135": self.mq135,
                "temperature": round(self.temperature, 1),
                "humidity": round(self.humidity, 1),
                "aqi": round(self.aqi, 1),
                "aqi_label": lbl,
                "aqi_color": f"rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},1)",
                "connected": self.connected,
                "last_ts": self.last_ts,
                "history": self.aqi_history[-60:],
            }


ds = DataStore()


# ── ESP Poller ───────────────────────────────────────────────────────────────
def esp_poll():
    while True:
        try:
            if NET_OK:
                req = urlreq.urlopen(f"{ESP_URL}/data", timeout=3)
                raw = json.loads(req.read().decode())
                ds.update(raw)
                ds.connected = True
            else:
                raise Exception("no network")
        except Exception:
            ds.connected = False
            ds.demo_tick()  # fall back to demo simulation
        time.sleep(POLL_SECS)


poller = threading.Thread(target=esp_poll, daemon=True)
poller.start()


# ── User Store ───────────────────────────────────────────────────────────────
class UserStore:
    def __init__(self):
        self._data = self._load()

    def _load(self):
        try:
            with open(USER_FILE) as f: return json.load(f)
        except: return {}

    def _save(self):
        with open(USER_FILE, "w") as f: json.dump(self._data, f, indent=2)

    def _hash(self, pw):
        return hashlib.sha256(pw.encode()).hexdigest()

    def register(self, email, pw, name):
        if email in self._data:
            return False, "Account already exists."
        self._data[email] = {"name": name, "email": email,
                              "pw": self._hash(pw),
                              "created": datetime.now().isoformat()}
        self._save()
        return True, "Account created!"

    def login(self, email, pw):
        u = self._data.get(email)
        if not u or u["pw"] != self._hash(pw): return False, None
        return True, {k: v for k, v in u.items() if k != "pw"}


users = UserStore()


# ── Recommendations ───────────────────────────────────────────────────────────
def get_recommendations(aqi, temp, hum):
    recs = []
    if aqi < 51:
        recs.append({"category": "OUTDOOR", "icon": "🏃", "title": "Great for a walk!",
                     "body": f"AQI is {aqi:.0f} (Excellent). Head outside and breathe deep.",
                     "color": "#10b981", "bg": "#d1fae5"})
        recs.append({"category": "HEALTH", "icon": "🛡", "title": "Air quality is Good",
                     "body": "No mask needed. Enjoy outdoor activities freely.",
                     "color": "#10b981", "bg": "#d1fae5"})
    elif aqi < 101:
        recs.append({"category": "HEALTH", "icon": "😷", "title": "Moderate air quality",
                     "body": "Sensitive individuals should limit prolonged outdoor exertion.",
                     "color": "#f59e0b", "bg": "#fef3c7"})
    elif aqi < 151:
        recs.append({"category": "HEALTH", "icon": "😷", "title": "Unhealthy for Sensitive Groups",
                     "body": "Reduce outdoor time. Consider N95 for extended outdoor stays.",
                     "color": "#f97316", "bg": "#ffedd5"})
    elif aqi < 201:
        recs.append({"category": "HEALTH", "icon": "⚠️", "title": "Unhealthy – Stay Alert",
                     "body": "Everyone may experience effects. Wear N95/KN95 outdoors. Keep windows closed.",
                     "color": "#ef4444", "bg": "#fee2e2"})
    else:
        recs.append({"category": "HEALTH", "icon": "🚨", "title": "HAZARDOUS – Stay Indoors",
                     "body": "Emergency. Sealed N95/P100 respirator required. Seal doors and windows.",
                     "color": "#7f1d1d", "bg": "#fecaca"})

    if temp > 30:
        recs.append({"category": "TEMPERATURE", "icon": "🌡", "title": "High Temperature Alert",
                     "body": f"It's {temp}°C. Avoid outdoor exertion 11AM–4PM. Drink 250ml water every 30 min.",
                     "color": "#f97316", "bg": "#ffedd5"})
    elif temp < 10:
        recs.append({"category": "TEMPERATURE", "icon": "🧥", "title": "Cold Weather",
                     "body": "Temperature is low. Wear warm layered clothing outdoors.",
                     "color": "#3b82f6", "bg": "#dbeafe"})

    if hum > 80:
        recs.append({"category": "HUMIDITY", "icon": "💧", "title": "Very High Humidity",
                     "body": f"Humidity at {hum:.0f}%. Use a dehumidifier and ensure adequate ventilation.",
                     "color": "#6366f1", "bg": "#e0e7ff"})
    elif hum < 30:
        recs.append({"category": "HUMIDITY", "icon": "💨", "title": "Low Humidity",
                     "body": "Air is dry. Use a humidifier and stay hydrated.",
                     "color": "#8b5cf6", "bg": "#ede9fe"})

    if aqi > 100 and temp > 30:
        recs.append({"category": "ALERT", "icon": "⚠️", "title": "Combined Health Risk",
                     "body": "High AQI + high temperature increases cardiovascular strain significantly.",
                     "color": "#dc2626", "bg": "#fee2e2"})

    recs.append({"category": "ENVIRONMENT", "icon": "🌿", "title": "Add a Snake Plant",
                 "body": "Boost indoor O₂ levels naturally. Snake plants efficiently filter indoor toxins.",
                 "color": "#059669", "bg": "#d1fae5"})

    return recs


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pw    = request.form.get("password", "")
        ok, user = users.login(email, pw)
        if ok:
            session["user"] = user
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        pw    = request.form.get("password", "")
        pw2   = request.form.get("password2", "")
        if not name or not email or not pw:
            error = "Please fill all fields."
        elif pw != pw2:
            error = "Passwords do not match."
        else:
            ok, msg = users.register(email, pw, name)
            if ok:
                return redirect(url_for("login"))
            error = msg
    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


@app.route("/predict")
def predict():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("predict.html", user=session["user"])


@app.route("/advice")
def advice():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("advice.html", user=session["user"])


# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/data")
def api_data():
    return jsonify(ds.snapshot())


@app.route("/api/predict")
def api_predict():
    snap = ds.snapshot()
    pred = load_and_predict(ds.mq2, ds.mq135, ds.temperature, ds.humidity, ds.aqi_history)
    lbl, _ = aqi_label(pred)
    h = ds.aqi_history[-20:]
    trend = "stable"
    if len(h) >= 3:
        d = h[-1] - h[-3]
        trend = "worsening" if d > 5 else ("improving" if d < -5 else "stable")
    return jsonify({"predicted_aqi": pred, "label": lbl, "trend": trend,
                    "history": h, "current_aqi": snap["aqi"]})


@app.route("/api/recommendations")
def api_recs():
    snap = ds.snapshot()
    recs = get_recommendations(snap["aqi"], snap["temperature"], snap["humidity"])
    return jsonify(recs)


@app.route("/api/model_meta")
def api_model_meta():
    try:
        with open("models/model_meta.json") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"status": "not trained"})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🌬  SmartAir — Atmospheric Intelligence Platform")
    print("="*55)
    print("\n🤖 Running ML model training first...")
    train_run()
    print("\n🚀 Starting Flask server on http://0.0.0.0:5000")
    print("   ESP32 target:", ESP_URL)
    print("   Demo mode active when ESP not reachable")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

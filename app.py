import os
import sqlite3
import hashlib
import secrets
import warnings
import functools
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, g
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "instance", "vuna.db")

# ─── Load Model & Data ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "optimised_stacking_adr_model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "drug_df.csv")

model = joblib.load(MODEL_PATH)
df    = pd.read_csv(DATA_PATH)

# Build lookup tables from dataset
DRUG_LIST     = sorted(df["drugname"].dropna().unique().tolist())
DRUG_ID_MAP   = df.groupby("drugname")["drug_id"].first().to_dict()
ALL_ADR_LIST  = sorted(df["pt"].dropna().unique().tolist())

# Extract encoded feature names from model
MODEL_FEATURES = list(model.feature_names_in_)

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            created  TEXT    NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            drug        TEXT    NOT NULL,
            age         REAL,
            sex         TEXT,
            result      TEXT    NOT NULL,
            probability REAL,
            created     TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─── Prediction Logic ─────────────────────────────────────────────────────────
def build_input_vector(drug: str, age: float, sex: str) -> pd.DataFrame:
    """Build the one-row DataFrame matching the model's expected features."""
    row = {feat: 0 for feat in MODEL_FEATURES}

    # Age
    row["age"] = age if age and not np.isnan(float(age)) else df["age"].median()

    # Drug one-hot
    key = f"drugname_{drug.lower()}"
    if key in row:
        row[key] = 1

    # role_cod (always PS in dataset)
    if "role_cod_PS" in row:
        row["role_cod_PS"] = 1

    # Sex
    if sex == "F" and "sex_F" in row:
        row["sex_F"] = 1
    elif sex == "M" and "sex_M" in row:
        row["sex_M"] = 1

    # Drug ID one-hot
    drug_id = DRUG_ID_MAP.get(drug.lower())
    if drug_id:
        key_id = f"drug_id_{drug_id}"
        if key_id in row:
            row[key_id] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]

def predict_adr(drug: str, age: float, sex: str):
    X = build_input_vector(drug, age, sex)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(max(proba)) * 100

    # ADR reactions known for this drug from the dataset
    drug_reactions = df[df["drugname"] == drug]["pt"].dropna().unique().tolist()

    result = "ADR Detected" if pred == 1 else "No ADR Detected"
    return {
        "result": result,
        "prediction": int(pred),
        "confidence": round(confidence, 2),
        "drug_reactions": drug_reactions[:10],  # top 10
        "drug": drug,
        "age": age,
        "sex": sex,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm]):
            flash("All fields are required.", "danger")
            return render_template("signup.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("signup.html")

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("Email already registered.", "danger")
            return render_template("signup.html")

        db.execute(
            "INSERT INTO users (name, email, password, created) VALUES (?, ?, ?, ?)",
            (name, email, hash_password(password), datetime.utcnow().isoformat())
        )
        db.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db   = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and verify_password(password, user["password"]):
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    recent = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created DESC LIMIT 5",
        (session["user_id"],)
    ).fetchall()
    total = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ?",
        (session["user_id"],)
    ).fetchone()[0]
    adr_count = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ? AND result = 'ADR Detected'",
        (session["user_id"],)
    ).fetchone()[0]
    return render_template(
        "dashboard.html",
        recent=recent,
        total=total,
        adr_count=adr_count,
        drug_count=len(DRUG_LIST),
    )


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result_data = None

    if request.method == "POST":
        drug = request.form.get("drug", "").strip().lower()
        age_raw = request.form.get("age", "").strip()
        sex = request.form.get("sex", "").strip()

        # Validate
        if drug not in DRUG_LIST:
            flash("Please select a valid drug from the list.", "danger")
            return render_template("predict.html", drugs=DRUG_LIST, result=None)

        try:
            age = float(age_raw) if age_raw else df["age"].median()
            if age < 0 or age > 120:
                raise ValueError
        except ValueError:
            flash("Please enter a valid age (0-120).", "danger")
            return render_template("predict.html", drugs=DRUG_LIST, result=None)

        if sex not in ("M", "F"):
            flash("Please select a valid sex.", "danger")
            return render_template("predict.html", drugs=DRUG_LIST, result=None)

        result_data = predict_adr(drug, age, sex)

        # Save to history
        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, drug, age, sex, result, probability, created) VALUES (?,?,?,?,?,?,?)",
            (
                session["user_id"], drug, age, sex,
                result_data["result"],
                result_data["confidence"],
                datetime.utcnow().isoformat()
            )
        )
        db.commit()

    return render_template("predict.html", drugs=DRUG_LIST, result=result_data)


# ── History ───────────────────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created DESC",
        (session["user_id"],)
    ).fetchall()
    return render_template("history.html", predictions=rows)


# ── Drug Info ─────────────────────────────────────────────────────────────────
@app.route("/drugs")
@login_required
def drugs():
    drug_info = []
    for drug in DRUG_LIST:
        reactions = df[df["drugname"] == drug]["pt"].dropna().unique().tolist()
        drug_info.append({
            "name": drug,
            "drug_id": DRUG_ID_MAP.get(drug, "N/A"),
            "reaction_count": len(reactions),
            "sample_reactions": reactions[:5],
        })
    return render_template("drugs.html", drugs=drug_info)


# ── API: drug autocomplete ─────────────────────────────────────────────────────
@app.route("/api/drugs")
@login_required
def api_drugs():
    q = request.args.get("q", "").lower()
    matches = [d for d in DRUG_LIST if q in d][:15]
    return jsonify(matches)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)

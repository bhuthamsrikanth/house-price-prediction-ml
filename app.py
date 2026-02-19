from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

# -----------------------------
# Load ML Model
# -----------------------------
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]

# -----------------------------
# Create Database
# -----------------------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sqft REAL,
            bedrooms REAL,
            bathrooms REAL,
            price REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sqft = float(request.form["sqft"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])

    features = np.array([[sqft, bedrooms, bathrooms]])
    price = model.predict(features)[0]

    # Save to database
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (sqft, bedrooms, bathrooms, price)
        VALUES (?, ?, ?, ?)
    """, (sqft, bedrooms, bathrooms, price))
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=round(price, 2))

@app.route("/history")
def history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    app.run()

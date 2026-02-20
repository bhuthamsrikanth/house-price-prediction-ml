# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()

    return render_template("index.html", rows=rows)


@app.route("/predict", methods=["POST"])
def predict():
    sqft = float(request.form["sqft"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])

    features = np.array([[sqft, bedrooms, bathrooms]])
    price = model.predict(features)[0]

    # Save to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (sqft, bedrooms, bathrooms, price)
        VALUES (?, ?, ?, ?)
    """, (sqft, bedrooms, bathrooms, price))
    conn.commit()

    # Fetch latest 5 records
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()

    return render_template("index.html",
                           prediction=round(price, 2),
                           rows=rows)
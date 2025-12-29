from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the saved model, scaler, encoder
data = pickle.load(open("fraud_model.pkl", "rb"))
model = data["model"]
encoder = data["encoder"]
scaler = data["scaler"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read values from form
    step = float(request.form["step"])
    trans_type = request.form["type"]
    amount = float(request.form["amount"])
    oldBalOrg = float(request.form["oldbalanceOrg"])
    newBalOrg = float(request.form["newbalanceOrig"])
    oldBalDest = float(request.form["oldbalanceDest"])
    newBalDest = float(request.form["newbalanceDest"])

    # Encode type column
    trans_type_encoded = encoder.transform([trans_type])[0]

    # Prepare input in SAME ORDER as training
    input_data = np.array([[step, trans_type_encoded, amount,
                            oldBalOrg, newBalOrg, oldBalDest,
                            newBalDest]])

    # APPLY SCALING
    scaled_data = scaler.transform(input_data)

    # Predict
    pred = model.predict(scaled_data)[0]

    result = "🚨 FRAUD DETECTED!" if pred == 1 else "✔ Legit Transaction"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)

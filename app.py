from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

clases = [
    "setosa",
    "versicolor",
    "virginica"
]

@app.route("/")
def home():

    return {
        "estado": "API funcionando correctamente"
    }

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    entrada = pd.DataFrame([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]])

    pred = model.predict(entrada)[0]

    return jsonify({
        "prediccion": clases[int(pred)]
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    prediction = model.predict([message])[0]

    if prediction == 1:
        result = "🚨 Spam Message"
    else:
        result = "✅ Not Spam"

    return render_template("index.html", prediction=result, message=message)


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")

with open(model_path, "rb") as f:
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

    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
# SMS Spam Classifier

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing and Logistic Regression.

## Live Demo

**Try it here:**
https://spam-prediction-t45q.onrender.com/

---

## Project Overview

This project uses **TF-IDF vectorization** and **Logistic Regression** to detect spam messages.

Users can enter any SMS-style text into the web interface, and the model predicts whether the message is:

* 🚨 **Spam**
* ✅ **Not Spam**

The project demonstrates a complete machine learning workflow:

* Data preprocessing
* Feature engineering
* Model training
* Evaluation
* Model serialization
* Web deployment using Flask + Gunicorn + Render

---

## Model Performance

### Final Evaluation Metrics

**F1 Score:**
0.926

### Confusion Matrix

[[969   1]
[19 126]]

### Interpretation

* **969** legitimate messages correctly classified
* **1** legitimate message incorrectly flagged as spam
* **126** spam messages correctly detected
* **19** spam messages missed

The model achieves:

* High precision
* Strong recall
* Very low false positive rate

---

## Machine Learning Pipeline

Message Input
↓
TF-IDF Vectorization
↓
Logistic Regression
↓
Prediction

---

## Technologies Used

### Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorizer

### Backend

* Flask
* Gunicorn

### Data Processing

* Pandas
* NumPy

### Deployment

* Render

---

## Project Structure

```bash
project/
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── Procfile
└── spam_data_set.csv
```

---

## How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train Model

```bash
python src/train.py
```

This generates:

```bash
models/best_model.pkl
```

---

### 4. Run Web App

```bash
python src/predict.py
```

Open:

http://127.0.0.1:5000/

---

## Downloading the Trained Model

The trained model is available inside:

```bash
models/best_model.pkl
```

To use it in another Python project:

```python
import pickle

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(["Congratulations! You won $500"])
print(prediction)
```

Output:

* `1` → Spam
* `0` → Not Spam

---

## Example Predictions

### Spam Example

Input:

Congratulations! You have won $1000. Click here now!

Prediction:

🚨 Spam

---

### Ham Example

Input:

Hey, are we still meeting at 7 PM today?

Prediction:

✅ Not Spam

---

## Key Learnings

This project helped explore:

* Text preprocessing
* Feature extraction using TF-IDF
* Binary classification
* Pipeline creation in Scikit-learn
* Model evaluation using F1-score and confusion matrix
* Deployment of ML models as web apps

---

## Future Improvements

Potential upgrades:

* Hyperparameter tuning with GridSearchCV
* LinearSVC comparison
* Probability confidence scores
* Improved frontend design
* REST API support

---

## Live Application

https://spam-prediction-t45q.onrender.com/

---


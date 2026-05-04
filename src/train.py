import pickle
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# load-dataset
df = pd.read_csv("sales.csv")
df['v1'] = df['v1'].map({"ham":0 , "spam":1})
X_train,X_test,y_train,y_test = train_test_split(df['v2'],df['v1'],test_size=0.2,random_state=100)

# pipeline
pipe = Pipeline([("tdidf",TfidfVectorizer(max_features = 1000 ,ngram_range=(1,2))),("lg", LogisticRegression())])
pipe.fit(X_train,y_train)

# prediction
y_pred = pipe.predict(X_test)

# score
f1score = f1_score(y_test,y_pred)  
confusion_score = confusion_matrix(y_test,y_pred) 

print("F1 Score:", f1score)
print("Confusion Matrix:")
print(confusion_score)

# Save model
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Model saved successfully.")
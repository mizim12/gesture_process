import pickle
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


DATASET_PATH = 'gestures_dataset.csv'
MODEL_PATH = os.path.join('..', 'models', 'gesture_model.pkl')
df = pd.read_csv(DATASET_PATH, header=None)

X = df.iloc[:,1:]
y = df.iloc[:, 0]

print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(acc)

print(classification_report(y_test, y_pred))

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

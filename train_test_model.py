import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, auc
import pickle

def load_df_from_hdf5(group):
    dset = group["features"]
    columns = dset.attrs["column_names"]
    df = pd.DataFrame(dset[...], columns=columns)
    return df

hdf5_filename = "dataset.hdf5"

# open h5 and the train/test groups
with h5py.File(hdf5_filename, "r") as h5f:
    train_group = h5f["Segmented data/Train"]
    test_group = h5f["Segmented data/Test"]
    
    train_df = load_df_from_hdf5(train_group)
    test_df = load_df_from_hdf5(test_group)

print("Training DataFrame shape:", train_df.shape)
print("Testing DataFrame shape:", test_df.shape)

non_feature_cols = ["Activity", "Person", "Run", "Unique_Run"]
feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

X_train = train_df[feature_cols].values
y_train = train_df["Activity"].values

X_test = test_df[feature_cols].values
y_test = test_df["Activity"].values

#logistic reg model training
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

#model eval.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_disp = ConfusionMatrixDisplay(cm).plot()
print("Confusion Matrix:")
print(cm)

#ROC curve
y_score = clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
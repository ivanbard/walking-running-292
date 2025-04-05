import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, auc
import pickle
from sklearn.model_selection import learning_curve, ShuffleSplit

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

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

#get learning curve
train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train, y_train, cv=cv, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

#calc mean for scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Accuracy", color="blue")
plt.fill_between(train_sizes,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.2, color="blue")
plt.plot(train_sizes, val_scores_mean, label="Validation Accuracy", color="green")
plt.fill_between(train_sizes,
                 val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std,
                 alpha=0.2, color="green")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()

with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
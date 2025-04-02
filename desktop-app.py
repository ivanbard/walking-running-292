import sys
from PyQt5.QtWidgets import (QLabel, QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
import pandas as pd
import pickle
import numpy as np
from collections import Counter
from scipy.stats import skew, kurtosis

def extract_features(df): #find features in segments
    features = {}
    axes = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    for axis in axes:
        if axis in df.columns:
            signal = df[axis].values
            
            features[f"{axis}_mean"] = np.mean(signal)
            features[f"{axis}_std"] = np.std(signal)
            features[f"{axis}_max"] = np.max(signal)
            features[f"{axis}_min"] = np.min(signal)
            features[f"{axis}_range"] = np.max(signal) - np.min(signal)
            features[f"{axis}_median"] = np.median(signal)
            features[f"{axis}_variance"] = np.var(signal)
            features[f"{axis}_skewness"] = skew(signal)
            features[f"{axis}_kurtosis"] = kurtosis(signal)
            features[f"{axis}_rms"] = np.sqrt(np.mean(signal**2))
    
    return features

def segment_run(df, window_duration=5): #obtain 5 sec segments 
    segments = []
    max_time = df["Time (s)"].max()
    start_time = 0
    while start_time < max_time:
        segment = df[(df["Time (s)"] >= start_time) & (df["Time (s)"] < start_time + window_duration)]
        if not segment.empty:
            segments.append(segment)
        start_time += window_duration
    return segments

def preprocess_df(df, window_size=5):
    if df.isnull().values.any():
        df = df.interpolate(method='linear')  # fill data using interpolation
    
    columns_to_smooth = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    for col in columns_to_smooth:  # moving average filter on each column
        if col in df.columns:
            df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean()

    df = df.reset_index(drop=True)
    return df

def preprocess_and_extract_features(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    df = preprocess_df(df, window_size=5)
    
    segments = segment_run(df, window_duration=5) #segment into 5s sections if time column is present
    
    feature_list = []
    for segment in segments:
        feats = extract_features(segment)
        if "Activity" in df.columns:
            feats["Activity"] = df["Activity"].iloc[0]
        if "Person" in df.columns:
            feats["Person"] = df["Person"].iloc[0]
        feature_list.append(feats)
    
    features_df = pd.DataFrame(feature_list)
    return features_df


def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Walking vs. Jumping Classifier")
        self.setMinimumSize(1000, 800) #min window size
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(20)
        self.central_widget.setLayout(self.layout)
        
        self.button = QPushButton("Select CSV File")
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 20px;
                padding: 15px;
                font-size: 25px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.layout.addWidget(self.button)

        instructions = QLabel()
        instructions.setText(
            "<h2>Welcome!</h2>"
            "<p>Please select a CSV file containing your sensor data. "
            "The application will segment the data into 5-second windows, extract features, "
            "and then classify the data as <b>Walking</b> or <b>Jumping</b>.</p>"
            "<p>Make sure your CSV contains a <i>Time (s)</i> column along with the sensor data.</p>"
            "<p>   </p>"
            "<p>You will be prompted to choose a location to save your output file after selecting"
            " a CSV. The programs output will be displayed after choosing a save location.</p>"
        )
        instructions.setWordWrap(True)
        self.layout.addWidget(instructions)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #808080;
            }
            QLabel {
                font-size: 24px;
                color: #2c3e50;
            }
        """)
        
        self.button.clicked.connect(self.select_file)

    def select_file(self):
        options = QFileDialog.Options() #opens file dialog to pick file
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.predict_file(file_name)
    
    def predict_file(self, file_path):
        try:
            features_df = preprocess_and_extract_features(file_path) #perform feature extraction
            if features_df.empty:
                QMessageBox.critical(self, "Error", "No features extracted. Check your CSV and processing code.")
                return
            
            #load trained model and its scaler
            model, scaler = load_model()
            
            X = features_df.values
            X_scaled = scaler.transform(X)
            
            #make window specific predictions
            predictions = model.predict(X_scaled)
            label_map = {0: "Walking", 1: "Jumping"}
            pred_labels = [label_map[p] for p in predictions]
            
            #to create the 'majority' of the prediction
            majority_vote = Counter(pred_labels).most_common(1)[0][0]

            #save output into a csv
            output_df = features_df.copy()
            output_df["Predicted_label"] = pred_labels
            save_file, _ = QFileDialog.getSaveFileName(
                self, "Save Predictions", "", "CSV Files (*.csv)"
            )
            if not save_file:
                save_file = "predictions_output.csv"
            
            output_df.to_csv(save_file, index=False)
            
            # output message box
            QMessageBox.information(
                self, "Prediction",
                f"Majority Prediction: {majority_vote}\n"
                f"Segment Predictions: {pred_labels}\n"
                f"Results saved to: {save_file}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
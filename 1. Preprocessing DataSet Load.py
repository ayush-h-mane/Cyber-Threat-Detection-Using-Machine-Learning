import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- NOTE: Replace 'your_data.csv' with the actual path to your combined/cleaned dataset ---
# A typical file from CIC-IDS2017 is named something like 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
try:
    df = pd.read_csv("C:\\College code\\Major Project\\DataSets\\TrafficLabelling\\Wednesday-workingHours.pcap_ISCX.csv") 
    print(f"Dataset loaded. Initial shape: {df.shape}")
except FileNotFoundError:
    print("Error: C:\\College code\\Major Project\\DataSets\\TrafficLabelling\\Wednesday-workingHours.pcap_ISCX.csv not found. Please replace with your actual file path.")
    # Exit or use a mock dataset if the actual file isn't available
    exit()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')  # Suppress XGBoost warnings

# === Load Data ===
df = pd.read_csv("water_dataX.csv", encoding='ISO-8859-1')

# === Rename & Convert Key Columns ===
df['pH'] = pd.to_numeric(df['PH'], errors='coerce')
df['Conductivity'] = pd.to_numeric(df['CONDUCTIVITY (µmhos/cm)'], errors='coerce')
df['Coliform'] = pd.to_numeric(df['TOTAL COLIFORM (MPN/100ml)Mean'], errors='coerce')

# === Drop Rows with Missing Values ===
df.dropna(subset=['pH', 'Conductivity', 'Coliform'], inplace=True)

# === Define Classification Rule ===
def classify_status(row):
    if 6.5 <= row['pH'] <= 8.5 and row['Conductivity'] <= 500 and row['Coliform'] <= 1000:
        return 'Good'
    else:
        return 'Poor'

df['Status'] = df.apply(classify_status, axis=1)
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# === 1. Count of Good vs Poor ===
plt.figure(figsize=(6, 4))
sns.countplot(x='Status', data=df, palette='Set2')
plt.title('Water Quality Status Count')
plt.xlabel('Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# === 2. Distribution Plots ===
plt.figure(figsize=(16, 4))

# pH
plt.subplot(1, 3, 1)
sns.histplot(df['pH'], kde=True, color='blue')
plt.title('pH Distribution')

# Conductivity
plt.subplot(1, 3, 2)
sns.histplot(df['Conductivity'], kde=True, color='green')
plt.title('Conductivity Distribution')

# Coliform
plt.subplot(1, 3, 3)
sns.histplot(df['Coliform'], kde=True, color='red')
plt.title('Coliform Distribution')

plt.tight_layout()
plt.show()

# === Encode Labels ===
label_encoder = LabelEncoder()
df['Status_encoded'] = label_encoder.fit_transform(df['Status'])  # Good=1, Poor=0 (usually)

# === Select Features ===
features = ['pH', 'Conductivity', 'Coliform']
X = df[features]
y = df['Status_encoded']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# === Predictions ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save Model ===
joblib.dump(model, "water_quality_model.pkl")
print("\n💾 Model saved as 'water_quality_model.pkl'")


# model.py

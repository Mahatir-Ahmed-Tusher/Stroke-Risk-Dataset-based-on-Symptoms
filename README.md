# Stroke Risk Dataset based on Symptoms


[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset)

## 📌 Overview

The **Stroke Risk Prediction Dataset** is a comprehensive dataset designed for machine learning and medical research purposes. It provides insights into various factors influencing stroke risk, allowing for binary classification (risk vs. no risk) and regression (risk percentage prediction). The dataset contains **70,000** samples with 18 critical health-related features.

This dataset aims to facilitate stroke prediction using machine learning, aiding early diagnosis and potential preventive measures.

## 📂 Dataset Details

- **Total Samples:** 70,000
- **Number of Features:** 18
- **Target Variables:**
  - `At Risk (Binary)`: Indicates whether an individual is at risk of a stroke (0 = No, 1 = Yes).
  - `Stroke Risk (%)`: Represents the estimated percentage risk of stroke.
- **Feature Categories:**
  - **Symptoms-related features** (e.g., Chest Pain, Dizziness, Nausea/Vomiting)
  - **Physiological conditions** (e.g., High Blood Pressure, Irregular Heartbeat)
  - **Lifestyle and risk factors** (e.g., Snoring/Sleep Apnea, Anxiety/Feeling of Doom)

## 📊 Column Descriptions

| Feature Name                    | Description |
|---------------------------------|-------------|
| `Chest Pain`                     | 1 if the individual experiences chest pain, else 0. |
| `Shortness of Breath`            | 1 if the individual experiences breathlessness, else 0. |
| `Irregular Heartbeat`            | 1 if an irregular heartbeat is present, else 0. |
| `Fatigue & Weakness`             | 1 if the individual frequently experiences fatigue or weakness, else 0. |
| `Dizziness`                      | 1 if dizziness is a symptom, else 0. |
| `Swelling (Edema)`               | 1 if swelling in limbs is observed, else 0. |
| `Pain in Neck/Jaw/Shoulder/Back`  | 1 if pain in these regions occurs, else 0. |
| `Excessive Sweating`             | 1 if excessive sweating is reported, else 0. |
| `Persistent Cough`               | 1 if a persistent cough is present, else 0. |
| `Nausea/Vomiting`                | 1 if nausea or vomiting is observed, else 0. |
| `High Blood Pressure`            | 1 if the person has hypertension, else 0. |
| `Chest Discomfort (Activity)`    | 1 if chest discomfort occurs during activity, else 0. |
| `Cold Hands/Feet`                | 1 if the individual often has cold extremities, else 0. |
| `Snoring/Sleep Apnea`            | 1 if sleep disturbances like snoring or apnea are present, else 0. |
| `Anxiety/Feeling of Doom`        | 1 if the individual experiences anxiety or a sense of impending doom, else 0. |
| `Stroke Risk (%)`                | Percentage risk of stroke (continuous value). |
| `At Risk (Binary)`               | 1 if the individual is at risk of stroke, else 0. |
| `Age`                            | Age of the individual in years. |

## 🏥 Data Provenance

### 📌 Sources

This dataset has been developed using references from established medical literature, including:
- **"Stroke: Pathophysiology, Diagnosis, and Management" by J.P. Mohr et al.**
- **"Oxford Textbook of Stroke and Cerebrovascular Disease" by H. Markus et al.**
- **American Heart Association (AHA) and Centers for Disease Control and Prevention (CDC) stroke risk factors guidelines.**

### 🛠️ Collection Methodology

The dataset was synthetically generated based on statistical distributions obtained from real-world medical studies. The dataset ensures a **50:50 distribution** between individuals at risk and not at risk, making it balanced for both classification and regression tasks.

To ensure accuracy, probability-weighted sampling was used, incorporating risk factor dependencies like age, high blood pressure, and symptom severity. **Feature engineering** has been performed to derive meaningful insights from the dataset, ensuring robustness for predictive modeling.

## 🔬 Possible Use Cases

This dataset can be used for:
- **Stroke Prediction Models:** Train machine learning models for stroke risk classification.
- **Health Risk Analysis:** Identify high-risk individuals based on symptoms and conditions.
- **Feature Importance Studies:** Analyze which factors contribute most to stroke risk.
- **Clinical Decision Support:** Develop AI-driven decision-making tools for healthcare professionals.

## 🚀 Getting Started

### 🔽 Download the Dataset
You can download the dataset directly from Kaggle:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset)

### 📌 Loading the Dataset in Python
```python
import pandas as pd

# Load dataset
file_path = "stroke_risk_dataset_70000.csv"
df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())
```

### 🔍 Exploratory Data Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of Stroke Risk (%)
sns.histplot(df['Stroke Risk (%)'], bins=30, kde=True, color='red')
plt.title('Distribution of Stroke Risk (%)')
plt.show()

# Countplot for At Risk (Binary)
sns.countplot(x='At Risk (Binary)', data=df, palette=['green', 'red'])
plt.title('At Risk (Binary) Distribution')
plt.show()
```

## 🤖 Machine Learning Implementation

### Example: Train a Logistic Regression Model
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define features and target
X = df.drop(columns=['At Risk (Binary)', 'Stroke Risk (%)'])
y = df['At Risk (Binary)']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 📜 Citation
If you use this dataset in your research or projects, please cite:

```
@dataset{tusher2025stroke,
  author    = {Mahatir Ahmed Tusher},
  title     = {Stroke Risk Prediction Dataset},
  year      = {2025},
  url       = {https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset}
}
```

## 📢 Contribution
Contributions are welcome! Feel free to raise issues or submit pull requests.

## 🔗 Connect with Me
- **GitHub**: [Mahatir-Ahmed-Tusher](https://github.com/Mahatir-Ahmed-Tusher)
- **LinkedIn**: [Mahatir Ahmed Tusher](https://in.linkedin.com/in/mahatir-ahmed-tusher-5a5524257)
- **Google Scholar**: [Mahatir Ahmed Tusher](https://scholar.google.com/citations?user=k8hhhx4AAAAJ&hl=en)

---

**📢 If you find this dataset useful, don't forget to star the repo and upvote on Kaggle!** ⭐


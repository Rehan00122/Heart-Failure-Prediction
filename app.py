
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Comprehensive CSS for background and black text
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
    font-family: 'Arial', sans-serif;
    color: #000000 !important;
}
.markdown-text-container, .stMarkdown, .stText, .stDataFrame, .stTable, .stSlider, .stSelectbox, .stButton, .stForm {
    color: #000000 !important;
}
.stSlider label, .stSelectbox label, .stButton>button, .stForm label {
    color: #000000 !important;
}
pre, code {
    color: #000000 !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Heart Disease Prediction Dashboard")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart.csv")
        df = df.astype({
            'Sex': 'string',
            'ChestPainType': 'string',
            'RestingECG': 'string',
            'ExerciseAngina': 'string',
            'ST_Slope': 'string'
        }, errors='ignore')
        return df
    except FileNotFoundError:
        st.error("Error: 'heart.csv' not found. Please ensure the file is in the correct directory.")
        return None

df = load_data()
if df is None:
    st.stop()

# Data Preview
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Dataset Info
st.subheader("Dataset Info")
st.write("Shape of dataset:", df.shape)
st.write("Column types:")
st.write(df.dtypes)

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Visualizations")

# Histograms
st.markdown("#### Histograms of Numeric Features")
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='#1e90ff')
    axes[i].set_title(f'Distribution of {col}', color='black')
    axes[i].set_xlabel(col, color='black')
    axes[i].set_ylabel('Count', color='black')
    axes[i].tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Box Plots
st.markdown("#### Box Plots of Numeric Features")
fig, ax = plt.subplots(figsize=(10, 6))
df[numeric_cols].boxplot(ax=ax)
ax.set_title('Box Plots for Numeric Features', color='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, color='black')
ax.set_ylabel('Value', color='black')
ax.tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Count Plots
st.markdown("#### Count Plots of Categorical Features")
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x=col, hue='HeartDisease', ax=ax, palette='Blues_d')
    ax.set_title(f'Count Plot of {col} by HeartDisease', color='black')
    ax.set_xlabel(col, color='black')
    ax.set_ylabel('Count', color='black')
    ax.tick_params(axis='both', colors='black')
    ax.legend(title='HeartDisease', labelcolor='black')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, color='black')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Correlation Heatmap
st.markdown("#### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax, annot_kws={'color': 'black'})
ax.set_title("Correlation Between Numeric Features", color='black')
ax.tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Scatter Plot: Age vs MaxHR
st.markdown("#### Age vs MaxHR by HeartDisease")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df, x='Age', y='MaxHR', hue='HeartDisease', style='HeartDisease', ax=ax, palette='Blues_d')
ax.set_title('Age vs MaxHR by HeartDisease', color='black')
ax.set_xlabel('Age', color='black')
ax.set_ylabel('MaxHR', color='black')
ax.tick_params(axis='both', colors='black')
ax.legend(title='HeartDisease', labelcolor='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Preprocessing and Model Training
st.subheader("Predictive Modeling")

# Preprocessing
model_df = df.copy()
# Handle zero values
for col in ['RestingBP', 'Cholesterol']:
    model_df[col] = model_df[col].replace(0, model_df[col][model_df[col] != 0].median())

# Encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    le_dict[col] = le

# Features and target
X = model_df.drop('HeartDisease', axis=1)
y = model_df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
st.markdown("#### Model Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
st.markdown(f"""
<pre style="color: #000000; background-color: #ffffff;">
{classification_report(y_test, y_pred)}
</pre>
""", unsafe_allow_html=True)

# Confusion Matrix
st.markdown("#### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], 
            yticklabels=['No Heart Disease', 'Heart Disease'], ax=ax, annot_kws={'color': 'black'})
ax.set_title('Confusion Matrix', color='black')
ax.set_xlabel('Predicted', color='black')
ax.set_ylabel('Actual', color='black')
ax.tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Feature Importance
st.markdown("#### Feature Importance")
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax, palette='Blues_d')
ax.set_title('Feature Importance', color='black')
ax.set_xlabel('Importance', color='black')
ax.set_ylabel('Feature', color='black')
ax.tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Interactive Prediction
st.subheader("Predict Heart Disease Risk")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 80, 50)
        resting_bp = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar (>120 mg/dL)", [0, 1])
    with col2:
        max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
        oldpeak = st.slider("Oldpeak (ST Depression)", -2.0, 6.0, 0.0, step=0.1)
        sex = st.selectbox("Sex", le_dict['Sex'].classes_)
        chest_pain = st.selectbox("Chest Pain Type", le_dict['ChestPainType'].classes_)
    col3, col4 = st.columns(2)
    with col3:
        resting_ecg = st.selectbox("Resting ECG", le_dict['RestingECG'].classes_)
        exercise_angina = st.selectbox("Exercise Angina", le_dict['ExerciseAngina'].classes_)
    with col4:
        st_slope = st.selectbox("ST Slope", le_dict['ST_Slope'].classes_)
    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input features
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [le_dict['Sex'].transform([sex])[0]],
        'ChestPainType': [le_dict['ChestPainType'].transform([chest_pain])[0]],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [le_dict['RestingECG'].transform([resting_ecg])[0]],
        'MaxHR': [max_hr],
        'ExerciseAngina': [le_dict['ExerciseAngina'].transform([exercise_angina])[0]],
        'Oldpeak': [oldpeak],
        'ST_Slope': [le_dict['ST_Slope'].transform([st_slope])[0]]
    })
    # Ensure input_data columns match X.columns
    input_data = input_data[X.columns]
    # Scale numeric features
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    # Predict
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        st.success(f"Predicted Heart Disease Risk: **{'Yes' if prediction == 1 else 'No'}** (Probability: {prediction_proba[1]:.2%})")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Key Insights
st.subheader("Key Insights")
st.markdown("""
- **Data Quality**: Zero values in RestingBP and Cholesterol were replaced with medians to handle potential errors.
- **Key Predictors**: Features like Oldpeak, ST_Slope, and ExerciseAngina are critical, as shown by feature importance.
- **Model Performance**: The Random Forest Classifier achieves balanced accuracy, precision, and recall, suitable for heart disease prediction.
- **Clinical Relevance**: High Oldpeak, Flat/Down ST_Slope, and ExerciseAngina (Y) are strongly associated with heart disease risk.
- **Interactive Tool**: Use the prediction form to estimate heart disease risk based on patient characteristics.
""")
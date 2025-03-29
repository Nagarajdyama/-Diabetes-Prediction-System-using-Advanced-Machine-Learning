import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

st.set_page_config(page_title=" Advanced Diabetes Prediction", layout="wide")

st.markdown(
    """
    <style>
    body {background-color: #0e0e0e; color: white; font-family: 'Poppins', sans-serif;}
    h1 {color: #00e6ff; text-align: center; font-size: 40px; font-weight: bold; text-shadow: 0px 0px 20px #00e6ff;}
    .stButton>button {border-radius: 10px; background: linear-gradient(90deg, #00e6ff, #0072ff); color: white;
                       font-size: 20px; padding: 12px 25px; border: none; transition: 0.3s ease-in-out;}
    .stButton>button:hover {box-shadow: 0px 0px 15px #00e6ff; transform: scale(1.05);}
    .stDataFrame {border-radius: 12px; background: #1e1e1e; color: white;}
    .sidebar .sidebar-content {background-color: #121212; color: white;}
    .metric-container {background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px;
                        text-align: center; font-size: 18px; box-shadow: 0px 0px 15px rgba(0, 255, 255, 0.3);}
    </style>
    """,
    unsafe_allow_html=True,
)

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    st.subheader(" Raw Data Preview")
    st.dataframe(df.head())
    return df


def visualize_data(df):
    st.subheader(" Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.write("####  Outcome Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df['Outcome'], palette='coolwarm', ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("####  Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)



def train_model(df):
    st.subheader(" Training the Model...")
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    expected_columns = X.columns.tolist()
    joblib.dump(expected_columns, 'expected_columns.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with st.spinner(" Training Model... Please wait!"):
        time.sleep(2)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f" Model Trained Successfully with Accuracy: {accuracy:.2f}")

    st.write("###  Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("###  Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
    st.pyplot(fig)
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')



def predict_diabetes(uploaded_file):
    st.subheader(" Predicting Diabetes...")
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding="utf-8")

    expected_columns = joblib.load('expected_columns.pkl')
    df = df[expected_columns]

    X = scaler.transform(df)
    predictions = model.predict(X)

    df['Prediction'] = predictions
    positive_cases = df[df['Prediction'] == 1]
    total_cases = len(df)
    positive_count = len(positive_cases)

    with st.spinner(" Analyzing results..."):
        time.sleep(2)
    st.success(f" {positive_count} people out of {total_cases} tested POSITIVE for diabetes.")
    st.dataframe(positive_cases)

    df.to_csv("diabetes_predictions.csv", index=False)
    positive_cases.to_csv("diabetes_positive_cases.csv", index=False)

    st.download_button(" Download All Predictions", "diabetes_predictions.csv", "text/csv")
    st.download_button(" Download Positive Cases", "diabetes_positive_cases.csv", "text/csv")

    return positive_cases


st.title("Advanced Diabetes Prediction System")

uploaded_file = st.file_uploader(" Upload a CSV file with patient data", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    visualize_data(df)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Train Model"):
            train_model(df)
    with col2:
        if st.button(" Predict Diabetes"):
            predict_diabetes(uploaded_file)

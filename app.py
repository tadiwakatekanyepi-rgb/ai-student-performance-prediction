# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Login successful! Navigate using sidebar.")
        else:
            st.error("Invalid credentials")
    st.stop()

# ---------------- THEME SELECTOR ----------------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    bg_color = "#121212"
    text_color = "#FFFFFF"
    bar_colors = {"Low": "green", "Medium": "orange", "High": "red"}
else:
    bg_color = "#f5f7fa"
    text_color = "#000000"
    bar_colors = {"Low": "green", "Medium": "orange", "High": "red"}

st.markdown(f"""
<style>
body {{background-color: {bg_color}; color: {text_color};}}
.stDataFrame div {{color: {text_color};}}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Data Upload & Prediction", "Risk Analysis & Recommendations"])

# ---------------- DATA UPLOAD & PREDICTION ----------------
if menu == "Data Upload & Prediction":
    st.title("📊 AI-Assisted Student Performance Prediction")
    uploaded_file = st.file_uploader("Upload CSV file (required)", type=["csv"])
    target_col = st.text_input("Target Column (e.g., Final_Grade)")

    if uploaded_file and target_col:
        df = pd.read_csv(uploaded_file)
        st.subheader("Sample Data")
        st.dataframe(df.head())

        # ---------------- PREPROCESS ----------------
        student_ids = df["Student_ID"] if "Student_ID" in df.columns else df.index
        X = df.drop(columns=[target_col, "Student_ID"], errors='ignore')
        y = df[target_col]

        # Encode categorical
        encoders = {}
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------------- MODEL TRAINING ----------------
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        svm = SVR()
        svm.fit(X_scaled, y)

        # ---------------- RISK CLASSIFICATION ----------------
        risk_labels = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled, risk_labels)
        predicted_risk = knn.predict(X_scaled)

        # ---------------- MODEL EVALUATION ----------------
        rf_cv = cross_val_score(rf, X_scaled, y, cv=3).mean()
        svm_cv = cross_val_score(svm, X_scaled, y, cv=3).mean()
        precision = precision_score(risk_labels, predicted_risk, average="weighted")
        f1 = f1_score(risk_labels, predicted_risk, average="weighted")

        st.subheader("📈 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RF Cross-Validation", f"{rf_cv:.2f}")
        col2.metric("SVM Cross-Validation", f"{svm_cv:.2f}")
        col3.metric("KNN Precision", f"{precision:.2f}")
        col4.metric("KNN F1 Score", f"{f1:.2f}")

        # ---------------- SAVE FOR DASHBOARD ----------------
        st.session_state.risk_df = pd.DataFrame({
            "Student_ID": student_ids,
            "Risk_Level": predicted_risk
        })
        st.session_state.X = X
        st.session_state.knn_model = knn
        st.session_state.rf_model = rf
        st.session_state.svm_model = svm
        st.session_state.scaler = scaler
        st.session_state.encoders = encoders
        st.session_state.y = y

        # ---------------- TABS ----------------
        tabs = st.tabs(["SHAP Feature Importance", "Risk Distribution", "Predict New Student"])

        # ---------- SHAP TAB ----------
        with tabs[0]:
            st.write("🧩 Feature Importance (SHAP)")
            sample_X = X.sample(min(100, len(X)), random_state=42)  # sample for speed
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(sample_X)
            shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
            plt.clf()

        # ---------- RISK DISTRIBUTION TAB ----------
        with tabs[1]:
            st.write("📊 Risk Distribution")
            fig = px.histogram(
                st.session_state.risk_df,
                x="Risk_Level",
                color="Risk_Level",
                color_discrete_map=bar_colors,
                title="Number of Students by Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------- PREDICT NEW STUDENT TAB ----------
        with tabs[2]:
            st.subheader("Predict Risk for a New Student")
            new_student_data = {}
            for col in X.columns:
                if col in st.session_state.encoders:
                    le = st.session_state.encoders[col]
                    new_student_data[col] = st.selectbox(f"{col}", le.classes_)
                else:
                    new_student_data[col] = st.number_input(f"{col}", value=0)
            if st.button("Predict New Student Risk"):
                new_df = pd.DataFrame([new_student_data])
                # Encode categorical
                for col in new_df.select_dtypes(include="object").columns:
                    le = st.session_state.encoders[col]
                    new_df[col] = le.transform(new_df[col])
                # Scale
                new_scaled = st.session_state.scaler.transform(new_df)
                # Predict
                new_risk = st.session_state.knn_model.predict(new_scaled)[0]
                st.success(f"Predicted Risk Level: {new_risk}")

# ---------------- RISK ANALYSIS & RECOMMENDATIONS ----------------
if menu == "Risk Analysis & Recommendations":
    st.title("⚠️ Students Risk Analysis & Recommendations")

    if "risk_df" not in st.session_state:
        st.warning("Upload data and run predictions first in 'Data Upload & Prediction' tab.")
        st.stop()

    risk_df = st.session_state.risk_df

    # ---------------- SIDEBAR FILTER ----------------
    st.sidebar.subheader("Filter Students by Risk Level")
    filter_risk = st.sidebar.multiselect(
        "Select Risk Levels", options=["Low","Medium","High"], default=["Low","Medium","High"]
    )
    filtered_df = risk_df[risk_df["Risk_Level"].isin(filter_risk)]

    # ---------------- RISK DISTRIBUTION ----------------
    st.subheader("Risk Level Distribution")
    fig = px.histogram(
        filtered_df,
        x="Risk_Level",
        color="Risk_Level",
        color_discrete_map=bar_colors,
        title="Filtered Students by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- AT-RISK STUDENTS ----------------
    st.subheader("Students at Selected Risk Levels")
    st.dataframe(filtered_df)

    # ---------------- RECOMMENDATIONS ----------------
    st.subheader("Recommendations")
    high_risk = filtered_df[filtered_df["Risk_Level"]=="High"]
    if not high_risk.empty:
        for idx, row in high_risk.iterrows():
            st.markdown(f"**Student {row['Student_ID']}**: Recommend tutoring, mentoring, counseling, and personalized learning support.")
    else:
        st.success("No students at high risk!")
        
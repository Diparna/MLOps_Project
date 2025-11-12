import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

page = st.sidebar.radio("Navigation", ["Prediction", "Monitoring"])

@st.cache_data
def load_data():
    df = pd.read_csv("df_clean.csv")  
    return df

df = load_data()

@st.cache_resource
def train_logreg_model(df):
    numeric_features = [
        'Temperature(F)', 'Visibility(mi)', 'Humidity(%)', 
        'Wind_Speed(mph)', 'Pressure(in)', 'Precipitation(in)', 'Distance(mi)'
    ]
    categorical_features = [
        'Weather_Group', 'visibility_bin', 'rush_hour', 
        'is_weekend', 'is_holiday', 'State'
    ]

    df['HighSeverity'] = (df['Severity'] > 2).astype(int)

    X = df[numeric_features + categorical_features]
    y = df['HighSeverity']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = encoder.fit_transform(X[categorical_features])
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    X_combined = pd.concat([
        pd.DataFrame(X[numeric_features].reset_index(drop=True)),
        pd.DataFrame(X_cat, columns=encoded_cols)
    ], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred)
    }
    return model, scaler, encoder, numeric_features, categorical_features, metrics

model, scaler, encoder, num_cols, cat_cols, model_metrics = train_logreg_model(df)

if page == "Prediction":
    st.header("Accident Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", sorted(df["State"].dropna().unique()))
        valid_cities = df.loc[df["State"] == state, "City"].dropna().unique()
        city = st.selectbox("City",sorted(valid_cities) if len(valid_cities) > 0 else ["No cities available"])
        day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].dropna().unique()))
        hour = st.slider("Hour of Day (24h)", 0, 23, 8)
        rush_hour = 1 if hour in [7,8,9,16,17,18] else 0
        is_weekend = 1 if day in ["Saturday","Sunday"] else 0
        is_holiday = st.selectbox("Holiday?", ["No","Yes"]) == "Yes"

    with col2:
        weather = st.selectbox("Weather", sorted(df["Weather_Group"].dropna().unique()))
        temp = st.number_input("Temperature (°F)", value=60.0)
        vis = st.slider("Visibility (mi)", 0.1, 10.0, 5.0)
        humidity = st.slider("Humidity (%)", 0, 100, 70)
        wind = st.slider("Wind Speed (mph)", 0, 50, 5)
        pressure = st.number_input("Pressure (inHg)", value=30.0)
        precip = st.number_input("Precipitation (in)", value=0.0)
        distance = st.number_input("Distance (mi)", value=0.5)
        vis_bin = "Low" if vis < 1 else "Medium" if vis < 5 else "High"
    sample = pd.DataFrame([{
        'Temperature(F)': temp,
        'Visibility(mi)': vis,
        'Humidity(%)': humidity,
        'Wind_Speed(mph)': wind,
        'Pressure(in)': pressure,
        'Precipitation(in)': precip,
        'Distance(mi)': distance,
        'Weather_Group': weather,
        'visibility_bin': vis_bin,
        'rush_hour': rush_hour,
        'is_weekend': is_weekend,
        'is_holiday': int(is_holiday),
        'State': state
    }])

    if st.button("🚦 Predict Accident Risk"):
        X_cat = encoder.transform(sample[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        X_combined = pd.concat([
            pd.DataFrame(sample[num_cols].reset_index(drop=True)),
            pd.DataFrame(X_cat, columns=encoded_cols)
        ], axis=1)
        X_scaled = scaler.transform(X_combined)

        prob = model.predict_proba(X_scaled)[:, 1][0]
        label = "High Risk" if prob > 0.6 else "Medium Risk" if prob > 0.3 else "Low Risk"

        st.success(f"**Predicted Accident Risk:** {label}")
        st.metric("Predicted Probability", f"{prob*100:.1f}%")
        st.info(f"{weather} weather, {'rush hour' if rush_hour else 'off-peak'}, {day} at {hour}:00 in {city}, {state}.")

elif page == "Monitoring":
    st.header("📈 Model Monitoring Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
    col2.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
    col3.metric("Records Used", len(df))

    st.subheader("Confusion Matrix")
    cm = model_metrics["confusion"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Feature Drift Check (Example)")
    feature = st.selectbox("Select feature to monitor:", ["Weather_Group", "Hour", "rush_hour", "is_weekend"])
    old = df[df["Year"] < 2022][feature].value_counts(normalize=True)
    new = df[df["Year"] >= 2022][feature].value_counts(normalize=True)
    drift_df = pd.concat([old.rename("Before"), new.rename("After")], axis=1).fillna(0)
    drift_df.plot(kind="bar", figsize=(8,4))
    plt.title(f"Feature Distribution Shift: {feature}")
    st.pyplot(plt)

    st.caption("If large shifts are seen in input features, consider retraining the model.")

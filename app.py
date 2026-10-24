
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="App Addiction Risk Predictor",
    page_icon="📱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("risk_model.pkl")

st.title("📱 App Addiction Risk Predictor")
st.write(
    "Explore how your app usage habits relate to "
    "potentially problematic usage patterns."
)

st.info(
    "Educational tool only. This prediction is not a "
    "medical diagnosis or a validated clinical assessment."
)

try:
    saved = load_model()
    model = saved["model"]
    features = saved["features"]
except (FileNotFoundError, KeyError, ValueError):
    st.error("Model not found or invalid. Run train_model.py first.")
    st.stop()

with st.form("risk_form"):
    st.subheader("Enter your usage habits")

    daily_screen_hours = st.slider(
        "Daily screen time (hours)", 0.0, 16.0, 4.0, 0.5
    )

    app_opens_per_day = st.number_input(
        "App openings per day", 0, 500, 50
    )

    social_media_hours = st.slider(
        "Social media use (hours/day)", 0.0, 12.0, 2.0, 0.5
    )

    night_usage_hours = st.slider(
        "Phone use late at night (hours)", 0.0, 8.0, 1.0, 0.5
    )

    sleep_hours = st.slider(
        "Average sleep (hours/night)", 0.0, 14.0, 8.0, 0.5
    )

    control_difficulty = st.slider(
        "Difficulty controlling app use (1 = low, 5 = high)",
        1, 5, 2
    )

    submitted = st.form_submit_button("Predict risk")

if submitted:
    user_data = {
        "daily_screen_hours": daily_screen_hours,
        "app_opens_per_day": app_opens_per_day,
        "social_media_hours": social_media_hours,
        "night_usage_hours": night_usage_hours,
        "sleep_hours": sleep_hours,
        "control_difficulty": control_difficulty
    }

    X_new = pd.DataFrame([user_data])[features]
    prediction = model.predict(X_new)[0]

    probabilities = model.predict_proba(X_new)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    st.subheader("Prediction result")

    if prediction == "Low":
        st.success("Estimated risk pattern: Low")
    elif prediction == "Moderate":
        st.warning("Estimated risk pattern: Moderate")
    else:
        st.error("Estimated risk pattern: High")

    st.write(
        "This is a model estimate based on the data used for training, "
        "not an assessment of whether you have an addiction."
    )

    st.subheader("Model output")
    for label in ["Low", "Moderate", "High"]:
        if label in class_probabilities:
            st.write(f"{label}: {class_probabilities[label]:.1%}")
            st.progress(float(class_probabilities[label]))

    st.subheader("Healthy usage suggestions")

    suggestions = []

    if daily_screen_hours > 6:
        suggestions.append(
            "Review your daily screen-time report and set realistic limits."
        )

    if night_usage_hours > 2:
        suggestions.append(
            "Consider a phone-free wind-down period before sleep."
        )

    if sleep_hours < 7:
        suggestions.append(
            "Review your sleep routine and aim for sufficient rest."
        )

    if control_difficulty >= 4:
        suggestions.append(
            "Try app timers, notification controls, or planned breaks."
        )

    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write(
            "Continue monitoring whether app use supports your "
            "sleep, daily responsibilities, and well-being."
        )
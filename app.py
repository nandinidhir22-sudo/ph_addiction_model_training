from pathlib import Path

import pandas as pd
import streamlit as st
from pathlib import Path
import pickle
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "phone_addiction_models.pkl"

with open(MODEL_PATH, "rb") as file:
    bundle = pickle.load(file)



@st.cache_resource
def load_models():
    # Load only the model file you created and trust.
    return pickle.load(MODEL_PATH)


bundle = load_models()
classifier = bundle["classifier"]
regressor = bundle["regressor"]
FEATURE_COLUMNS = bundle["feature_columns"]
ADDICTION_CUTOFF = bundle["addiction_cutoff"]


st.title("Teen Phone Addiction Estimates")
st.write(
    "Enter the information below to get an addiction-status classification "
    "and an estimated Addiction_Level score."
)
st.caption(
    f"For this example, scores of {ADDICTION_CUTOFF} or above are labeled "
    "'Addicted'. This is an educational cutoff, not a clinical definition."
)


with st.form("prediction_form"):
    st.subheader("Teen information")

    age = st.number_input("Age", min_value=13, max_value=19, value=16, step=1)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    school_grade = st.selectbox(
        "School grade",
        ["7th", "8th", "9th", "10th", "11th", "12th"],
    )

    daily_usage = st.number_input(
        "Daily phone usage (hours)", 0.0, 24.0, 5.0, step=0.1
    )
    sleep_hours = st.number_input(
        "Sleep (hours)", 0.0, 24.0, 6.5, step=0.1
    )
    academic_performance = st.slider(
        "Academic performance", 0, 100, 70
    )
    social_interactions = st.slider(
        "Social interactions (0 to 10)", 0, 10, 5
    )
    exercise_hours = st.number_input(
        "Exercise (hours)", 0.0, 24.0, 1.0, step=0.1
    )

    anxiety_level = st.slider("Anxiety level (1 to 10)", 1, 10, 5)
    depression_level = st.slider("Depression level (1 to 10)", 1, 10, 5)
    self_esteem = st.slider("Self-esteem (1 to 10)", 1, 10, 5)

    parental_control = st.selectbox(
        "Parental control",
        [0, 1],
        format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
    )

    screen_before_bed = st.number_input(
        "Screen time before bed (hours)", 0.0, 24.0, 1.0, step=0.1
    )
    phone_checks = st.number_input(
        "Phone checks per day", min_value=0, value=50, step=1
    )
    apps_used = st.number_input(
        "Apps used daily", min_value=0, value=10, step=1
    )

    social_media_hours = st.number_input(
        "Time on social media (hours)", 0.0, 24.0, 2.0, step=0.1
    )
    gaming_hours = st.number_input(
        "Time on gaming (hours)", 0.0, 24.0, 1.0, step=0.1
    )
    education_hours = st.number_input(
        "Time on education apps (hours)", 0.0, 24.0, 1.0, step=0.1
    )

    usage_purpose = st.selectbox(
        "Main phone usage purpose",
        ["Browsing", "Other", "Education", "Social Media", "Gaming"],
    )
    family_communication = st.slider(
        "Family communication (1 to 10)", 1, 10, 5
    )
    weekend_usage = st.number_input(
        "Weekend phone usage (hours)", 0.0, 24.0, 6.0, step=0.1
    )

    submitted = st.form_submit_button("Get predictions")


if submitted:
    # Names here must match the dataset's feature column names exactly.
    input_values = {
        "Age": age,
        "Gender": gender,
        "School_Grade": school_grade,
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep_hours,
        "Academic_Performance": academic_performance,
        "Social_Interactions": social_interactions,
        "Exercise_Hours": exercise_hours,
        "Anxiety_Level": anxiety_level,
        "Depression_Level": depression_level,
        "Self_Esteem": self_esteem,
        "Parental_Control": parental_control,
        "Screen_Time_Before_Bed": screen_before_bed,
        "Phone_Checks_Per_Day": phone_checks,
        "Apps_Used_Daily": apps_used,
        "Time_on_Social_Media": social_media_hours,
        "Time_on_Gaming": gaming_hours,
        "Time_on_Education": education_hours,
        "Phone_Usage_Purpose": usage_purpose,
        "Family_Communication": family_communication,
        "Weekend_Usage_Hours": weekend_usage,
    }

    # Keep columns in the exact order used when training.
    input_df = pd.DataFrame([input_values]).reindex(
        columns=FEATURE_COLUMNS
    )

    predicted_status = int(classifier.predict(input_df)[0])
    predicted_score = float(regressor.predict(input_df)[0])

    st.subheader("Predictions")

    if predicted_status == 1:
        st.success("Classification: Addicted")
    else:
        st.info("Classification: Not addicted")

    st.write(
        f"Estimated Addiction_Level: **{predicted_score:.2f}** "
        "(the Linear Regression estimate can fall outside the original 1–10 range)"
    )

    st.caption(
        "These are model estimates from the provided dataset, not a medical assessment."
    )
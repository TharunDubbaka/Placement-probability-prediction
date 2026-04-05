import streamlit as st
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="IMG_20241113_134021.jpg",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    rf = joblib.load("C://Users//dubba//OneDrive//Desktop//Machine learning projects//student_placement_predictor//models//lgmodel.pkl")
    lg = joblib.load("C://Users//dubba//OneDrive//Desktop//Machine learning projects//student_placement_predictor//models//lgmodel.pkl")
    return rf, lg

rfmodel, lgmodel = load_models()

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>
.card {
    background: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("AI Placement Predictor")
st.markdown("### Enter your details to predict placement chances")

st.write("---")

# =========================
# FORM (BEST UX)
# =========================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Academic Profile")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
        internships = st.number_input("Internships", min_value=0, value=1)
        projects = st.number_input("Projects Completed", min_value=0, value=2)
        certifications = st.number_input("Certifications", min_value=0, value=1)

    with col2:
        st.subheader(" Skills & Performance")
        coding_hours = st.number_input("Coding Hours / Week", min_value=0, value=10)
        aptitude = st.number_input("Aptitude Score", min_value=0, max_value=100, value=65)
        communication = st.selectbox("Communication Skills", [1,2,3,4,5])
        hackathons = st.number_input("Hackathons", min_value=0, value=1)

    submit = st.form_submit_button(" Predict Placement")

# =========================
# FEATURE ENGINEERING
# =========================
def create_features():
    skill_score = (coding_hours*0.4 + aptitude*0.4 + communication*10*0.2)
    experience_score = projects + internships + hackathons

    return np.array([[cgpa, internships, coding_hours, projects,
                      certifications, communication, hackathons,
                      aptitude, skill_score, experience_score]])

# =========================
# PREDICTION
# =========================
if submit:

    sample = create_features()

    try:
        rf_prob = rfmodel.predict_proba(sample)[0]
        lg_prob = lgmodel.predict_proba(sample)[0]

        rf_score = rf_prob[1] + 0.5 * rf_prob[0]
        lg_score = lg_prob[1] + 0.5 * lg_prob[0]

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(" Random Forest")
            st.metric("Placement Chance", f"{rf_score*100:.2f}%")
            st.progress(float(rf_score))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(" Logistic Regression")
            st.metric("Placement Chance", f"{lg_score*100:.2f}%")
            st.progress(float(lg_score))
            st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # INTERPRETATION
        # =========================
        def interpret(score):
            if score > 0.75:
                return " High Chance"
            elif score > 0.5:
                return " Moderate Chance"
            else:
                return " Low Chance"

        st.write("---")
        st.subheader(" Insights")

        st.success(f"Random Forest: {interpret(rf_score)}")
        st.info(f"Logistic Regression: {interpret(lg_score)}")

        # =========================
        # SUGGESTIONS
        # =========================
        st.write("---")
        st.subheader(" Suggestions")

        if coding_hours < 15:
            st.warning("Increase coding practice")

        if aptitude < 70:
            st.warning("Improve aptitude ")

        if internships == 0:
            st.warning("Get internship experience ")

        if communication < 3:
            st.warning("Improve communication")

        if projects < 2:
            st.warning("Build more projects")

    except Exception as e:
        st.error(f"Prediction error: {e}")
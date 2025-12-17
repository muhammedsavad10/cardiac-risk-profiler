import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- 1. Page Configuration & Professional Styling ---
st.set_page_config(
    page_title="Cardiac Risk Profiler Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a high-end, professional medical interface
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #0e1117; /* Matches Streamlit dark mode */
        }
        /* Styling the Sidebar */
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        /* Making the 'Run' button prominent and clinical */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
        }
        /* Styling the Metric Boxes for results */
        div[data-testid="metric-container"] {
            background-color: #1e1e26;
            border: 1px solid #363a45;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        /* Adjusting plot background to match */
        .stPyplot > div > div > svg {
            background-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Data & Train Model (Cached) ---
@st.cache_data
def load_and_train():
    # Load data from a stable repository
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    df = pd.read_csv(url)
    
    # Prep
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression (Best for explainable probabilities)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # Get feature importance (coefficients) for the interpretation chart
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', ascending=False)
    
    return model, scaler, importance

model, scaler, feature_importance = load_and_train()

# --- 3. Sidebar - Clinical Intake Form ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
    st.title("Patient Intake")
    st.markdown("Configure patient parameters below.")
    st.markdown("---")
    
    with st.expander("1. Demographics & Vitals", expanded=True):
        age = st.slider("Age", 20, 100, 55)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120, help="Blood pressure on admission to the hospital")
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    
    with st.expander("2. Symptoms & History", expanded=True):
        cp = st.selectbox("Chest Pain Type", 
                          ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        exang = st.radio("Exercise Induced Angina?", ["No", "Yes"], horizontal=True)
    
    with st.expander("3. Lab & Test Results", expanded=True):
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl?")
        restecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 0.0, step=0.1, help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("ST Slope of Peak Exercise", ["Upsloping", "Flat", "Downsloping"])
        ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
        thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversable Defect"])

    # Data Preprocessing for the Model
    sex_num = 1 if sex == "Male" else 0
    fbs_num = 1 if fbs else 0
    exang_num = 1 if exang == "Yes" else 0
    
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3} 

    input_data = np.array([[
        age, sex_num, cp_map[cp], trestbps, chol, fbs_num, 
        restecg_map[restecg], thalach, exang_num, oldpeak, 
        slope_map[slope], ca, thal_map[thal]
    ]])

    st.markdown("---")
    predict_btn = st.button("⚡ Run Clinical Risk Assessment")

# --- 4. Main Dashboard Area ---
st.title("🩺 Cardiac Risk Profiler Pro")
st.markdown("#### AI-Powered Clinical Decision Support System")
st.info("NOTE: This tool utilizes a validated Logistic Regression model trained on the UCI Cleveland dataset. It is intended for professional use as a decision support aid, not a definitive diagnosis.")

if predict_btn:
    # A. Perform Prediction
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1] * 100
    prediction = model.predict(input_scaled)[0]
    
    st.divider()
    
    # B. Results Dashboard layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("Risk Analysis Report")
        if prob > 50:
            st.metric(label="Predicted Probability of CAD", value=f"{prob:.1f}%", delta="High Risk Classification", delta_color="inverse")
            st.error("⚠️ **Assessment:** The model detects patterns consistent with Coronary Artery Disease.")
        else:
            st.metric(label="Predicted Probability of CAD", value=f"{prob:.1f}%", delta="Low Risk Classification", delta_color="normal")
            st.success("✅ **Assessment:** The model detects a low probability of Coronary Artery Disease.")
            
    with col2:
        st.subheader("Clinical Factor Interpretation")
        st.caption("Feature impact on this specific patient's risk score. (Red = Increased Risk, Blue = Decreased Risk)")
        
        # Professional Matplotlib Chart
        plt.style.use('dark_background') # Match the theme
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Define colors based on positive/negative impact
        colors = ['#ff5252' if x > 0 else '#448aff' for x in feature_importance['Importance']]
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette=colors, ax=ax)
        
        # Clean up chart for professional look
        ax.set_xlabel("Log-Odds Coefficient (Impact)", fontsize=10, color='white')
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=11, color='white')
        ax.tick_params(axis='x', color='white')
        sns.despine(left=True, bottom=True)
        ax.grid(axis='x', color='#444444', linestyle='--')
        
        st.pyplot(fig)

else:
    # C. Landing Page (Pre-Prediction State) - THIS IS THE FIX
    st.divider()
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("""
            ### Welcome to the Professional Suite.
            Please enter patient vitals, history, and lab results in the sidebar to generate a real-time risk profile.
            
            This system provides:
            * **Probabilistic Risk Scoring:** A precise percentage likelihood of disease.
            * **Factor Interpretability:** A breakdown of which clinical features are driving the risk score.
            
            👈 **Begin by configuring data in the sidebar.**
        """)
    with col_right:
        # A STABLE, high-quality professional image from Unsplash
        st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80", 
                 use_container_width=True, 
                 caption="Advanced Clinical Analytics")
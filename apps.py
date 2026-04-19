import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

# LOAD MODEL
model_package = joblib.load("customer_segmentation_model.joblib")

preprocessor = model_package["preprocessor"]
kmeans = model_package["kmeans"]
prof_freq = model_package["prof_freq"]
var_freq = model_package["var_freq"]

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# Load CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Input Page
st.markdown("""
<div class="header">
    <h1><span class="icon">🚀</span> Customer Segment Prediction</h1>
    <p>AI-Powered Insights for Smarter Customer Targeting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="input-card">
        <h3><span class="icon">📊</span> Customer Profile</h3>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

#Left Column
with col1:
        st.markdown("#### 👤 Personal Info")
        gender = st.selectbox("👤 Gender",  ["Male", "Female"])
        ever_married = st.selectbox("👤 Ever Married", ["Yes", "No"])
        graduated = st.selectbox("👤 Graduated", ["Yes", "No"])
        age = st.slider("🎂 Age", 18, 80, 30)

#Right Column
with col2:
        st.markdown("#### 💼 Professional Info")
        profession = st.selectbox("💼 Profession", [
            "Healthcare", "Engineer", "Artist", "Doctor",
            "Marketing", "Executive", "Lawyer", "Entertainment"
        ])
        work_exp = st.slider("💼 Work Experience", 0, 20, 2)
        spending = st.selectbox("💰 Spending Score", ["Low", "Average", "High"])
        family_size = st.slider("👨‍👩‍👧 Family Size", 1, 10, 3)


var_1 = st.selectbox("Category (Var_1)", [
        "Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6", "Cat_7"
    ])


# Spacer columns center button
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
        predict_clicked = st.button("🚀 Predict Segment", use_container_width=True)

if predict_clicked:

    # Encoding binary features
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

    prof_val = np.log1p(prof_freq.get(profession, 0))
    var_val = np.log1p(var_freq.get(var_1, 0))

    input_data = pd.DataFrame([{
        "Gender": binary_map[gender],
        "Ever_Married": binary_map[ever_married],
        "Graduated": binary_map[graduated],
        "Age": age,
        "Work_Experience": work_exp,
        "Family_Size": family_size,
        "Profession_freq": prof_val,
        "Var_1_freq": var_val,
        "Spending_Score": spending
    }])

    # Predict
    try:
        X = preprocessor.transform(input_data)
        cluster = kmeans.predict(X)[0]
        distances = kmeans.transform(X)
        distance = np.min(distances)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        cluster = None

    # Display Results
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction Result")

    # ROW 1 - CORE INFO
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    
    colA.markdown(f'''
    <div class="metric-box">
        <div class="metric-title"><span class="icon">🎯</span> Cluster</div>
        <div class="metric-value">{cluster}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    colB.markdown(f'''
    <div class="metric-box">
        <div class="metric-title"><span class="icon">🎂</span> Age</div>
        <div class="metric-value">{age}</div>
    </div>
    ''', unsafe_allow_html=True)

    colC.markdown(f'''
    <div class="metric-box">
        <div class="metric-title"><span class="icon">💼</span> Profession</div>
        <div class="metric-value">{profession}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2 - FEATURE ENGINEERING
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 🧠 Feature Insights")
    colA, colB = st.columns(2)

    # transform biar lebih readable
    prof_score = round(np.log1p(prof_val), 2)
    var_score = round(np.log1p(var_val), 2)

    # PROFESSION KPI
    colA.markdown(f'''
    <div class="metric-box">
        <div class="metric-title"><span class="icon">💼</span> Profession Score</div>
        <div class="metric-value">{prof_score}</div>
    </div>
    ''', unsafe_allow_html=True)

    # CATEGORY KPI
    colB.markdown(f'''
    <div class="metric-box">
        <div class="metric-title"><span class="icon">📊</span> Category Score</div>
        <div class="metric-value">{var_score}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)   

    # Output
    if cluster is not None:

        segment_descriptions = {
            0: "👨‍👩‍👧 Budget Family Customers",
            1: "💼 Premium Potential Customers",
            2: "🌱 Young Emerging Customers",
            3: "🏆 Established Professionals"
        }

        segment_name = segment_descriptions.get(cluster, "Unknown Segment")

        # Persona interpretation
        st.markdown("## 📌 Customer Persona", unsafe_allow_html=True)

        # STYLE OUTPUT 
        if cluster == 0:
            st.markdown(f"""
            <div class="persona-card persona-0">
                <div class="persona-title">👨‍👩‍👧 Budget Family</div>
                    Low spending, large household<br>
                    • Price sensitive<br>
                    • Needs bundle offers<br>
                    • Focus on affordability
            </div>
            """, unsafe_allow_html=True)

        elif cluster == 1:
            st.markdown(f"""
            <div class="persona-card persona-1">
                <div class="persona-title">💼 Premium Customer</div>
                    High potential & loyalty<br>
                    • Strong purchasing power<br>
                    • Suitable for premium offers<br>
                    • Loyalty program target
            </div>
            """, unsafe_allow_html=True)

        elif cluster == 2:
            st.markdown(f"""
            <div class="persona-card persona-2">
                <div class="persona-title">🌱 Emerging Customer</div>
                    Young & growing segment<br>
                        • Increasing spending trend<br>
                        • Good for engagement<br>
                        • Upselling opportunity
            </div>
            """, unsafe_allow_html=True)

        elif cluster == 3:
            st.markdown(f"""
            <div class="persona-card persona-3">
                <div class="persona-title">🏆 Established Professional</div>
                    Stable & high-value customer<br>
                    • Consistent income<br>
                    • Premium product fit<br>
                    • Long-term value
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="persona-card persona-error">
                <div class="persona-title">⚠️ Unable to Determine Segment</div>
                    We couldn't confidently assign this customer to a segment<br>
                    Please check the input data and try again.
            </div>
            """, unsafe_allow_html=True)

if predict_clicked:

    PRIMARY_COLOR = "#5ae4b1"
    SECONDARY_COLOR = "#29c3ce"

    colors = [PRIMARY_COLOR, SECONDARY_COLOR, "#2cb3bd"]

    col1, col2 = st.columns(2)

    # BAR CHART (LEFT)
    with col1:
        st.markdown("## 📊 Customer Profile")

        profile_df = pd.DataFrame({
                "Feature": ["Experience","Family"],
                "Value": [work_exp, family_size]
        })

        fig_bar = px.bar(profile_df, x="Feature", y="Value", text="Value")

        fig_bar.update_traces(
                marker=dict(
                        color=colors,
                        line=dict(color=SECONDARY_COLOR, width=1)
                ),
                textposition='outside'
        )

        fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=PRIMARY_COLOR),
                margin=dict(l=20, r=20, t=40, b=20)
        )
    
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # RADAR CHART (RIGHT)
    with col2:
        st.markdown("## 🕸️ Customer Radar")

        spending_map = {
            "Low": 0.3,
            "Average": 0.6,
            "High": 1.0
        }
    
        radar_fig = go.Figure()
    
        radar_fig.add_trace(go.Scatterpolar(
            r=[work_exp/20, family_size/10, spending_map[spending]],
            theta=["Experience","Family", "Spending"],
            fill='toself',
            fillcolor='rgba(59,72,221,0.25)',
            line=dict(color=PRIMARY_COLOR, width=3)
        ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1])
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=SECONDARY_COLOR),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

#Footer
st.markdown("""
<div class="footer">

<div class="footer-title">
🚀 Customer Segmentation Dashboard
</div>

<div class="footer-sub">
Machine Learning • Data Science • K-Means Clustering • Streamlit App
</div>

<div>
<a href="https://github.com/atikahdr" target="_blank">🐙 GitHub</a>
<span class="footer-divider">|</span>
<a href="https://share.streamlit.io/user/atikahdr" target="_blank">🌐 Live Demo</a>
</div>

<br>

<div style="font-size:12px; color:#9ca3af;">
Developed by <b>Atikah Dwi Rizky</b> • 2026
</div>

</div>
""", unsafe_allow_html=True)

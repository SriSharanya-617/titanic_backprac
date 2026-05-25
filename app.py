import streamlit as st
import math
import plotly.express as px
import pandas as pd

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Titanic AI Dashboard",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

.main {
    background-color:#0f172a;
}

h1,h2,h3,h4,h5,h6,p,label {
    color:white;
}

img {
    border-radius:15px;
}

.stButton>button {
    width:100%;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover {
    background:#1d4ed8;
    color:white;
}

.metric-card {
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 0px 10px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE SECTION
# ------------------------------------------------

st.markdown("""
<h1 style='text-align:center; color:#38bdf8;'>
🚢 Titanic Survival Prediction Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:#cbd5e1;'>
Artificial Neural Network Based Passenger Survival Analysis
</p>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HERO IMAGE
# ------------------------------------------------

st.image(
    "https://images.unsplash.com/photo-1518546305927-5a555bb7020d",
    use_container_width=True
)

# ------------------------------------------------
# TOP SECTION
# ------------------------------------------------

left, right = st.columns([1,1])

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------

with left:

    st.markdown("## Passenger Information")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.slider(
        "Age",
        1,
        80,
        24
    )

    fare = st.slider(
        "Fare",
        0.0,
        600.0,
        120.0
    )

    predict = st.button("Predict Survival")

# ------------------------------------------------
# PROJECT DESCRIPTION
# ------------------------------------------------

with right:

    st.markdown("## Model Overview")

    st.info("""
    This AI application predicts whether a passenger
    would survive during the Titanic disaster using
    a lightweight Artificial Neural Network (ANN).

    ### Features Used
    - Passenger Class
    - Age
    - Fare

    ### Deep Learning Concepts
    - Forward Propagation
    - Sigmoid Activation
    - Weighted Neural Computation
    """)

# ------------------------------------------------
# NORMALIZATION
# ------------------------------------------------

x1 = (pclass - 1) / 2
x2 = (age - 1) / 79
x3 = fare / 600

# ------------------------------------------------
# INITIAL WEIGHTS
# ------------------------------------------------

# Input -> Hidden

w_x1_h1 = 0.11
w_x2_h1 = 0.14
w_x3_h1 = 0.17

w_x1_h2 = 0.21
w_x2_h2 = 0.24
w_x3_h2 = 0.27

# Hidden Biases

b_h1 = 0.1
b_h2 = 0.1

# Hidden -> Output

w_h1_o1 = 0.31
w_h2_o1 = 0.34

# Output Bias

b_o1 = 0.1

# ------------------------------------------------
# SIGMOID FUNCTION
# ------------------------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ------------------------------------------------
# PREDICTION LOGIC
# ------------------------------------------------

if predict:

    # Hidden Layer

    net_h1 = (
        (x1 * w_x1_h1) +
        (x2 * w_x2_h1) +
        (x3 * w_x3_h1) +
        b_h1
    )

    net_h2 = (
        (x1 * w_x1_h2) +
        (x2 * w_x2_h2) +
        (x3 * w_x3_h2) +
        b_h2
    )

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    # Output Layer

    net_o1 = (
        (h1 * w_h1_o1) +
        (h2 * w_h2_o1) +
        b_o1
    )

    output = sigmoid(net_o1)

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    result = "Survived" if output > 0.5 else "Not Survived"

    survive_prob = output * 100
    nonsurvive_prob = 100 - survive_prob

    # ------------------------------------------------
    # METRICS SECTION
    # ------------------------------------------------

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction Result",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{output:.4f}"
        )

    with c3:
        st.metric(
            "Confidence Score",
            f"{survive_prob:.2f}%"
        )

    # ------------------------------------------------
    # PROGRESS BAR
    # ------------------------------------------------

    st.markdown("### Survival Confidence")

    st.progress(float(output))

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    st.markdown("---")
    st.markdown("## Prediction Visualization")

    chart1, chart2 = st.columns(2)

    df = pd.DataFrame({
        "Status": ["Survived", "Not Survived"],
        "Probability": [survive_prob, nonsurvive_prob]
    })

    # ------------------------------------------------
    # DONUT CHART
    # ------------------------------------------------

    with chart1:

        fig1 = px.pie(
            df,
            names="Status",
            values="Probability",
            hole=0.55
        )

        fig1.update_layout(
            height=350,
            paper_bgcolor="#0f172a",
            font_color="white"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # ------------------------------------------------
    # BAR CHART
    # ------------------------------------------------

    with chart2:

        fig2 = px.bar(
            df,
            x="Status",
            y="Probability",
            text="Probability"
        )

        fig2.update_layout(
            height=350,
            paper_bgcolor="#0f172a",
            font_color="white"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.caption("Built using Streamlit and Artificial Neural Networks")
import streamlit as st
import math
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# HEADER SECTION
# -----------------------------------

st.markdown(
    """
    <h1 style='text-align:center; color:#1E90FF;'>
    Titanic Survival Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:gray;'>
    Deep Learning Based Passenger Survival Prediction
    </h4>
    """,
    unsafe_allow_html=True
)

st.image(
    "https://cdn-icons-png.flaticon.com/512/3063/3063822.png",
    width=120
)

# -----------------------------------
# PROJECT DESCRIPTION
# -----------------------------------

st.markdown("## Project Description")

st.write("""
This application predicts whether a passenger would survive during the Titanic disaster
using an Artificial Neural Network (ANN).

The prediction is calculated using:
- Passenger Class
- Age
- Fare

The ANN uses:
- Forward Propagation
- Sigmoid Activation
- Backpropagation Logic
""")

# -----------------------------------
# INPUT SECTION
# -----------------------------------

st.markdown("## Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

# -----------------------------------
# NORMALIZATION
# -----------------------------------

# Manual Min-Max Scaling

x1 = (pclass - 1) / (3 - 1)
x2 = (age - 1) / (80 - 1)
x3 = (fare - 0) / (600 - 0)

# -----------------------------------
# INITIAL WEIGHTS
# -----------------------------------

# Input -> Hidden

w_x1_h1 = 0.11
w_x2_h1 = 0.14
w_x3_h1 = 0.17

w_x1_h2 = 0.21
w_x2_h2 = 0.24
w_x3_h2 = 0.27

# Hidden biases

b_h1 = 0.1
b_h2 = 0.1

# Hidden -> Output

w_h1_o1 = 0.31
w_h2_o1 = 0.34

# Output bias

b_o1 = 0.1

# -----------------------------------
# SIGMOID FUNCTION
# -----------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict Survival"):

    # -----------------------------------
    # FORWARD PROPAGATION
    # -----------------------------------

    # Hidden Layer

    net_h1 = (x1 * w_x1_h1) + \
             (x2 * w_x2_h1) + \
             (x3 * w_x3_h1) + b_h1

    net_h2 = (x1 * w_x1_h2) + \
             (x2 * w_x2_h2) + \
             (x3 * w_x3_h2) + b_h2

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    # Output Layer

    net_o1 = (h1 * w_h1_o1) + \
             (h2 * w_h2_o1) + b_o1

    output = sigmoid(net_o1)

    # -----------------------------------
    # PREDICTION LOGIC
    # -----------------------------------

    if output > 0.5:
        result = "Survived"
    else:
        result = "Not Survived"

    confidence = output * 100

    # -----------------------------------
    # OUTPUT SECTION
    # -----------------------------------

    st.markdown("## Prediction Output")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            label="Prediction Result",
            value=result
        )

    with col5:
        st.metric(
            label="Survival Probability",
            value=f"{output:.4f}"
        )

    with col6:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------

    st.markdown("## Probability Visualization")

    survive_prob = output
    not_survive_prob = 1 - output

    fig, ax = plt.subplots()

    ax.bar(
        ["Survived", "Not Survived"],
        [survive_prob, not_survive_prob]
    )

    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # Pie Chart

    fig2, ax2 = plt.subplots()

    ax2.pie(
        [survive_prob, not_survive_prob],
        labels=["Survived", "Not Survived"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)
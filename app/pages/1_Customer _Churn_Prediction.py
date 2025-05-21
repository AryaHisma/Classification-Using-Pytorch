import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib
import torch

st.set_page_config(page_title="Customer Churn Prediction", 
                   page_icon="📈",
                   layout='wide'
                   )

st.markdown(
    """
    <h3 style='text-align: center;'>🧾 Use Case: Customer Churn Prediction</h3>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns([4, 6])

with col1:
    
    st.markdown(
    """
    <h4 style='text-align: center;'>Description</h4>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 10px;'>

    <p>
    One powerful example of classification using neural networks is <strong>customer churn prediction</strong>.
    </p>

    <p>
    Businesses often ask:<br>
    <strong>"Which customers are likely to stop using our service?"</strong>
    </p>

    <p>
    By training a neural network on customer behavior data, we can model the likelihood of churn based on features such as:
    </p>

    <ul>
        <li>🕒 <strong>Tenure:</strong> Duration of customer relationship</li>
        <li>📞 <strong>Support Calls:</strong> Number of times contacting customer service</li>
        <li>💳 <strong>Payment Behavior:</strong> History of delayed payments or subscription type</li>
        <li>📈 <strong>Usage Patterns:</strong> Frequency and recency of product usage</li>
    </ul>

    <p>
    The model provides a <strong>probability score</strong> for each customer, allowing proactive strategies like offering promotions, improving support, or launching targeted retention campaigns.
    </p>

    <p>
    Compared to traditional methods like logistic regression, neural networks offer greater flexibility and can model more complex patterns in the data, especially when feature interactions are non-linear.
    </p>

    <p>
    This app enables you to explore this use case and beyond — using either the built-in example model or your own dataset.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


    
    
    
with col2:
    st.markdown(
    """
    <h4 style='text-align: center;'>Customer Churn Prediction</h3>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown(
    """
    **Input Data :**
    """
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        InputAge = st.number_input("Input Age", value=42, key="Input Age")
        
    with col2:
        InputTenure = st.number_input("Input Tenure", value=58, key="Input Tenure")
    
    with col3:
        InputUsageFrequency = st.number_input("Input Usage Frequency", value=1, key="Input Usage Frequency")
        
    with col4:
        InputSupportCalls = st.number_input("Input Support Calls", value=4, key="Input Support Calls")
        
        
    col5, col6, col7, col8 = st.columns(4) 
    
    with col5:
        InputPaymentDelay = st.number_input("Input Payment Delay", value=9, key="Input Payment Delay")
    
    with col6:
        InputTotalSpend = st.number_input("Input Total Spend", value=600, key="Input Total Spend")
    
    with col7:
        InputLastInteraction = st.number_input("Input Last Interaction", value=2, key="Input Last Interaction")
        
    with col8:
        gender = st.selectbox(
            "Input Gender",
            options=["Male", "Female"],  # atau ambil dari dataset
            key="input_gender"
        )
        
           
    col9, col10 = st.columns(2) 
    
    with col9:
        subscript = st.selectbox(
            "Subscription Type",
            options=["Standard", "Basic", "Premium"],  # atau ambil dari dataset
            key="input_Subscription Type"
        )
        
    with col10:
        contract_length = st.selectbox(
            "Contract Length",
            options=["Annual", "Monthly", "Quarterly"],  # atau ambil dari dataset
            key="input_Contract Length"
        )
        
        
    st.markdown(
    """
    **Prediksi :**
    """
    )
    
    input_data = {
    'Age' : InputAge,
    'Gender' : gender,
    'Tenure' : InputTenure,
    'Usage Frequency' : InputUsageFrequency,
    'Support Calls' : InputSupportCalls,
    'Payment Delay' : InputPaymentDelay,
    'Subscription Type' : subscript,
    'Contract Length' : contract_length,
    'Total Spend' : InputTotalSpend,
    'Last Interaction' : InputLastInteraction
    }

    st.write('Data Input :')
    input_df = pd.DataFrame(input_data, index=[0])
    st.write(input_df)

    
    # st.write('Preprocessing Data :')
    # Load Preprocessed Data
    pipeline_loaded = joblib.load('model/binaryclass/preprocessing_pipeline.pkl')
    
    
    # Transform Input Data
    input_processed = pipeline_loaded.transform(input_df)
    
    # st.write('Data Input :')
    # Ubah menjadi tensor
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)
    
    # Prediksi dengan model
    # Load Model
    model3 = torch.load('model/binaryclass/model_churn3.pth', weights_only=False)

    
    # Prediksi
    with torch.no_grad():
        output = model3(input_tensor)

    pred_prob = output.item()
    pred_label = 1 if pred_prob >= 0.5 else 0
    formatted_prob = f"{pred_prob:.2%}"

    st.markdown("### 🔍 Hasil Prediksi")

    if pred_label == 1:
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 10px; background-color: #ffe6e6'>
            <h4 style='color: #b30000'>⚠️ Prediksi: <strong>CHURN</strong></h4>
            <p>🎯 Probabilitas pelanggan akan churn: <strong>{formatted_prob}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 10px; background-color: #e6f9e6'>
            <h4 style='color: #006600'>✅ Prediksi: <strong>TIDAK CHURN</strong></h4>
            <p>🎯 Probabilitas pelanggan tetap loyal: <strong>{formatted_prob}</strong></p>
        </div>
        """, unsafe_allow_html=True)


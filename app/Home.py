# ---Library---
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Classifier using Neural Networks",
    page_icon="🧠",
    layout='wide'
)


st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 30px;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    <div class="centered-title">🧠 Neural Network Classifier</div>
    <div class="subtitle">Perform binary and multi-class classification using neural network — no coding required</div>
    """,
    unsafe_allow_html=True
)

st.markdown("### 👋 Welcome!")
st.markdown("""
This app allows you to explore and train **neural network-based classification models**. You can either use a **ready-to-use trained model**, or **build your own model** with your own tabular dataset.
""")

st.markdown(
    """
    <div class="highlight">
    <h4>⚙️ Key Features:</h4>
    <ul>
        <li>🤖 <strong>Try a Sample Model:</strong> Explore a trained model to see how it works in action.</li>
        <li>📁 <strong>Upload Your Dataset:</strong> Train a neural network model using your own CSV file (tabular format).</li>
        <li>🧠 <strong>Flexible Architecture:</strong> Build simple or deeper models — adjust layers, activation functions, and more.</li>
        <li>🔄 <strong>Binary & Multi-class Support:</strong> Automatically handles classification for both binary and multi-class targets.</li>
        <li>📊 <strong>Model Evaluation:</strong> Visualize performance metrics including accuracy, confusion matrix, and more.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="highlight">
        <h4>❓ What Can You Do With It?</h4>
        <p>
            This tool is ideal for:
        </p>
        <ul>
            <li>🧪 <strong>Any tabular classification task</strong></li>
        </ul>
        <p>
            With support for both binary and multi-class classification, you can apply it to a wide range of real-world datasets.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="highlight">
    <h4>🚀 Ready to Get Started?</h4>
    <p>Use the sidebar to choose your path:</p>
    <ul>
        <li>🏠 <strong>Home:</strong> This introduction page</li>
        <li>🔍 <strong>Explore Example Use Case (Customer Churn Prediction):</strong> See how the model works with a sample dataset</li>
        <li>💻 <strong>Build Your Own Model:</strong> Upload your own CSV and train a model with a few clicks</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)




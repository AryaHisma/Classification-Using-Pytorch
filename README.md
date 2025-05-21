# 🧠 Neural Network Classifier Web App

Welcome to an interactive and beginner-friendly deep learning application — powered by **Streamlit** and **PyTorch**.

This app lets you **train, evaluate, and deploy neural network models** for classification tasks — all without writing a single line of code!

---

## 👋 Welcome!

This app allows you to explore and train neural network-based classification models. You can either use a **ready-to-use trained model**, or build your own model with your own tabular dataset.

---

## ⚙️ Key Features

✅ **Explore Example Use Case (Customer Churn Prediction)**  
Start with a real-world example to understand how the model works using actual customer data.

📁 **Upload Your Own Dataset**  
Train a neural network model using your own CSV file (tabular format). Works with most classification datasets.

🧠 **Flexible Architecture**  
Easily adjust layers, activation functions, hidden units, and more to experiment with different neural net designs.

🔄 **Binary & Multi-class Support**  
Handles classification for both binary (e.g. churn vs no churn) and multi-class (e.g. iris dataset, etc).

📊 **Model Evaluation & Visualization**  
Includes loss curves, confusion matrix, accuracy scores, and more to help understand model performance.

---

## ❓ What Can You Do With It?

This tool is ideal for:

🧪 **Any tabular classification task**  
From customer churn or product category classification.

🎓 **Learning PyTorch & Deep Learning**  
Perfect for students, data analysts, and ML beginners who want to learn by doing — no coding required.

📈 **Business and Data Projects**  
Quickly prototype and deploy classification models to extract insights from structured data.

---

## 🗂 Project Structure

classification-using-pytorch/
├── app/
│   ├── Home.py
│   ├── function.py
│   ├── pages/
│   │   ├── 1_Customer_Churn_Prediction.py
│   │   ├── 2_Build_Your_model_Binary_Classification.py
│   │   └── 3_Build_Your_model_Multi_Classification.py
│   ├── model/
│   │   ├── binaryclass/
│   │   │   ├── model_churn3.pth
│   │   │   └── preprocessing_pipeline.pkl
│   │   └── multiclass/
│   │       ├── model_churn3.pth
│   │       └── preprocessing_pipeline.pkl
│   └── data/
├── requirements.txt
└── README.md

---

## 💻 How to Run the App Locally

```bash
# Clone the repository
git clone https://github.com/your-username/classification-using-pytorch.git
cd classification-using-pytorch

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/Home.py

👨‍💻 Created By
Made with ❤️ by Arya Hisma Maulana
Feel free to connect, fork, or contribute!
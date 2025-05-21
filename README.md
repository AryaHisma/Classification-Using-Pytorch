# 🧠 Customer Churn Prediction with PyTorch & Streamlit

🚀 A Streamlit web app for predicting customer churn using a neural network built with PyTorch. This project demonstrates the full pipeline: from data preprocessing and model training, to interactive predictions via a user-friendly web interface.

![Streamlit Screenshot](https://github.com/AryaHisma/Classification-Using-Pytorch/blob/main/picture/home.png)

---

## 📌 Features

- Upload your own dataset (`.csv`) for churn prediction
- Choose between pre-trained model or train a new one
- Visualize model evaluation: accuracy, confusion matrix, loss chart
- Handles binary and multi-class classification
- Built with modular, reusable code using PyTorch & Scikit-Learn

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend/ML**: PyTorch, scikit-learn, pandas, joblib
- **Visualization**: seaborn, matplotlib
- **Deployment**: Streamlit Cloud

---

## 📂 Project Structure

```
classification-using-pytorch/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Customer_Churn_Prediction.py
│       └── 2_Other_Feature_Page.py
│
├── model/
│   └── binaryclass/
│       └── preprocessing_pipeline.pkl
│
├── utils/
│   └── model_utils.py
│
├── requirements.txt
├── README.md
└── ...
```

---

## 🚀 Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/classification-using-pytorch.git
cd classification-using-pytorch
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app/Home.py
```

---

## 📈 Sample Use Case

- A telecom company wants to identify customers likely to churn.
- By uploading customer data, they get real-time prediction with explanation.
- Helps in targeted retention campaigns and revenue protection.

---

## 💡 Future Enhancements

- Feature importance visualization
- SHAP-based model interpretability
- Model explainability dashboard
- Integrate with database (e.g. PostgreSQL)

---

## 🙌 Credits

Made with ❤️ by [Arya Hisma Maulana](https://github.com/AryaHisma)

---

## 🪪 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

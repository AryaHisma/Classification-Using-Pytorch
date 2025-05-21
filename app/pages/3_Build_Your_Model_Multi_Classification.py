# ---Library---
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from function import *
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Build Your Own Model - Binari Classification", 
                   page_icon="🔢",
                   layout='wide'
                   )

st.markdown("""
    <h3 style='text-align: center;'>🧾 Build Your Own Model Classification using Neural Network</h3>
    <h5 style='text-align: center;'>Multi Classification</h5>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Upload Dataset CSV :**", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.markdown(
    """
    <h3 style='text-align: center;'>Preprocessing Data</h3>
    """,
    unsafe_allow_html=True
    )
    
    
    col1, col2 = st.columns([2, 8])
    
    with col1:
        df = df.dropna()
        
        st.write("")
        st.write("")
        st.write("")
        st.write(f"**Missing value dihapus otomatis**")
        st.write(f"Bentuk data : {df.shape}")
        
        
        
    
    with col2:
        st.write("**Dataset Preview**", df.head())
        
        
    col3, col4 = st.columns([2, 8])

    with col3:
        st.write("")
        st.write("")
        st.write("")
        
        drop = st.multiselect("**Select Target Drop :**", options=df.columns.tolist())
        
    with col4:
        df = df.drop(columns=drop)
    
        st.write("**Dataset Preview**", df.head())
        
    st.markdown(
    """
    <h3 style='text-align: center;'>Create Model</h3>
    """,
    unsafe_allow_html=True
    )
    
    st.write("**Transform Data :**")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        all_columns = df.columns.tolist()
        target = st.selectbox("**Select Target Column :**", options=all_columns)
        
        # Label Encoding species jadi angka 0,1,2
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        
        st.write(f"**Mapping Encode Data Target :**")
        # Lihat mapping: label -> angka
        for i, label in enumerate(le.classes_):
            st.write(f"{label} -> {i}")
        
    with col6:
        feature_cols = [col for col in all_columns if col != target]

        numerical_col = st.multiselect("Select Numerical Columns:", options=feature_cols)
    
    with col7:
        feature_cols = [col for col in all_columns if col != target]
        
        cat_onehot_col = st.multiselect("Select One Hot Columns:", options=feature_cols)
    
    with col8:
        feature_cols = [col for col in all_columns if col != target]
        
        cat_ordinal_col = st.multiselect("Select Ordinal Columns:", options=feature_cols)

    
    # Tentukan fitur dan label
    X = df.drop(columns=[target])
    y = df[target]


    # Tentukan Kolom
    numerical_cols = numerical_col
    categorical_onehot_cols = cat_onehot_col
    categorical_ordinal_cols = cat_ordinal_col

    # Buat list transformers secara dinamis
    transformers = []

    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))

    if categorical_onehot_cols:
        transformers.append(('onehot', OneHotEncoder(drop='first'), categorical_onehot_cols))

    if categorical_ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(), categorical_ordinal_cols))
        
    # Buat preprocessor   
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_col),
        ('onehot', OneHotEncoder(drop='first'), cat_onehot_col),
        ('ordinal', OrdinalEncoder(), cat_ordinal_col)
    ])

    # Buat Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Split Data Train Test
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        stratify=y, 
                                                        random_state=42)
    
    # Fit Transform
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    
    # Cek apakah onehot dan ordinal ada
    transformers = pipeline.named_steps['preprocessor'].transformers_

    # Mengecek apakah 'onehot' dan 'ordinal' tersedia di dalam transformers
    has_onehot = any(name == 'onehot' for name, _, _ in pipeline.named_steps['preprocessor'].transformers_)
    has_ordinal = any(name == 'ordinal' for name, _, _ in pipeline.named_steps['preprocessor'].transformers_)


    # One-hot
    if has_onehot and categorical_onehot_cols:
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['onehot']
        onehot_feature_names = ohe.get_feature_names_out(categorical_onehot_cols)
    else:
        onehot_feature_names = []

    # Ordinal
    if has_ordinal and categorical_ordinal_cols:
        ordinal_encoder = pipeline.named_steps['preprocessor'].named_transformers_['ordinal']
        categories = ordinal_encoder.categories_
        ordinal_col_info = dict(zip(categorical_ordinal_cols, categories))
    else:
        ordinal_encoder = None
        ordinal_col_info = {}

    # Gabungkan semua fitur akhir
    final_feature_names = (
        numerical_cols +
        list(onehot_feature_names) +
        categorical_ordinal_cols  # tetap tambahkan kolom nama ordinal meskipun tidak ditampilkan nilainya
    )

    
    # Tampilkan di kolom
    col9, col10 = st.columns(2)

    with col9:
        st.write("**The Sequence of The Transformed Column Names :**")
        st.text(final_feature_names)

    with col10:
        if ordinal_col_info:
            st.write("**Transformed Ordinal Encoder :**")
            for col, cat in ordinal_col_info.items():
                st.write(f"**{col}**:")
                for i, val in enumerate(cat):
                    st.markdown(f"- `{val}` → `{i}`")
        else:
            st.info("Tidak ada kolom dengan ordinal encoding.")
            
    
    try:
        X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
        
    except:
        st.error("""
                **Kesalahan input!**  
                Silakan periksa hal-hal berikut:
                - Refresh ulang halaman browser anda.
                - Pastikan data yang tidak diperlukan sudah dihapus manual atau menggunakan menu "Select Target Drop".
                - Pastikan Kolom "Select Target Column" sudah diubah manual menjadi angka.
                - Pastikan Kolom "Select Target Column' hanya berisi satu kolom saja.
                - Pastikan Kolom "Select Numerical Column" , "Select One Hot Categorical Column" , "Select Ordinal Categorical Column" diinput dengan benar.
                """)

    num_classes = len(np.unique(y_train))

    st.write("**Layer Setting :**")
    
    col11, col12 = st.columns(2)
    
    with col11:
        batch_size = st.number_input("Batch Size", value=32)
        
    with col12:
        num_layers = st.number_input("Hidden Layers", value=1)
        
        
    # Model parameter
    model_param = st.write("**Model Parameter :**")
    
    col13, col14, col15, col16 = st.columns(4)
    
    with col13:
        # Allow user setting hyperparameters
        epochs = st.number_input("Select Number of Epochs", value=10, key='epochs')
        
    with col14:
        # Dynamic layer configuration
        lr = st.number_input("Learning Rate", value=0.01, key='lr')
    
    with col15:  
        # Streamlit: Pilihan Loss Function
        loss_choice = st.selectbox("📉 Pilih Loss Function", ["CrossEntropyLoss"])
        st.markdown("""
        📌 **Catatan:**
        - `CrossEntropyLoss` digunakan untuk **multi-class classification** (output pakai argmax).
        """)

    with col16:
        # Streamlit: Pilihan Optimizer
        optimizer_choice = st.selectbox("⚙️ Pilih Optimizer", ["Adam", "SGD", "RMSprop"])
        
        
    # Layers
    layers = st.write("**Layer Configuration :**")  

    layers = []
    for i in range(num_layers):
        with st.expander(f"Layer {i+1} Configuration", expanded=True):
            units = st.slider(f"Units in Layer {i+1}", 16, 256, 64, key=f'units_{i}')
            activation = st.selectbox(
                f"Activation untuk Layer {i+1}", 
                ["relu", "tanh", "selu"], 
                key=f'act_{i}')
            
        in_features = X_train_tensor.shape[1] 
        # if i == 0 else layers[-2].out_features
        
        layers.append(nn.Linear(in_features, units))
        
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'selu':
            layers.append(nn.SELU())

    last_units = layers[-2].out_features if len(layers) >= 2 else units
    
    layers.append(nn.Linear(last_units, num_classes))

    model = nn.Sequential(*layers)
    
    
    st.write("**Model Architecture :**")
    st.code(str(model))
    
    
    

    if st.button("Training & Testing Model"):
        try:
            # Train model
            with st.spinner("Running ... please wait ⏳"):
                # Inisialisasi Loss Function
                criterion = nn.CrossEntropyLoss()
                
                # Inisialisasi Optimizer
                if optimizer_choice == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                elif optimizer_choice == "SGD":
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                elif optimizer_choice == "RMSprop":
                    optimizer = optim.RMSprop(model.parameters(), lr=lr)

                # Training Loop
                torch.manual_seed(42)
                
                losses_train = []
                losses_test = []

                for epoch in range(epochs):
                    model.train()
                    
                    output_train = model(X_train_tensor)
                    
                    loss_train = criterion(output_train, y_train_tensor)
                    
                    losses_train.append(loss_train.item())
                    
                    _, output_label_train = torch.max(output_train, 1)
            
                    accuracy_train = (output_label_train == y_train_tensor).float().mean()

                    optimizer.zero_grad()
                    
                    loss_train.backward()
                    
                    optimizer.step()

                    # Evaluasi akurasi
                    from sklearn.metrics import confusion_matrix, classification_report
                    
                    model.eval()
                    with torch.no_grad():
                        output_test = model(X_test_tensor)
                        
                        loss_test = criterion(output_test, y_test_tensor)
                        
                        losses_test.append(loss_test.item())
            
                        _, output_label_test = torch.max(output_test, 1)
                        
                        accuracy_test = (output_label_test == y_test_tensor).float().mean()
                        
                        if (epoch + 1) % 10 == 0:
                            st.write(f"Epoch : [{epoch+1}/{epochs}] | Loss_train : {loss_train.item():.4f} | Loss_test : {loss_test.item():.4f} | Accuracy_train : {accuracy_train:.4f}  | Accuracy Test : {accuracy_test:.4f}")


                # Plot loss per epoch menggunakan Matplotlib
                fig, ax = plt.subplots()
                ax.plot(range(epochs), losses_train, label='Train Loss')
                ax.plot(range(epochs), losses_test, label='Test Loss')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Loss per Epoch")
                ax.legend()
                ax.grid(True)

                # Tampilkan di Streamlit
                st.pyplot(fig)

                with torch.no_grad():
                    _, pred_train = torch.max(model(X_train_tensor), 1)
                    _, pred_test = torch.max(model(X_test_tensor), 1)

            cm_train = confusion_matrix(y_train, pred_train.numpy())
            cm_test = confusion_matrix(y_test, pred_test.numpy())

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Confusion Matrix - Train**")
                fig1, ax1 = plt.subplots()
                sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
                st.pyplot(fig1)
            with col2:
                st.write("**Confusion Matrix - Test**")
                fig2, ax2 = plt.subplots()
                sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
                st.pyplot(fig2)

            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Classification Report - Train**")
                st.code(classification_report(y_train, pred_train.numpy()))

            with col4:
                st.write("**Classification Report - Test**")
                st.code(classification_report(y_test, pred_test.numpy()))

            
            # Download model
            import joblib
            import io
            import base64
            
            # Simpan model ke buffer
            model_buffer = io.BytesIO()
            torch.save(model, model_buffer)
            model_buffer.seek(0)

            # Simpan preprocessor ke buffer
            preprocessor_buffer = io.BytesIO()
            joblib.dump(pipeline, preprocessor_buffer)
            preprocessor_buffer.seek(0)

            
            # Konversi ke base64 untuk membuat link download
            model_b64 = base64.b64encode(model_buffer.read()).decode()
            preprocessor_b64 = base64.b64encode(preprocessor_buffer.read()).decode()

            st.markdown("### 📥 Download Trained Assets")
            st.markdown(f'<a href="data:application/octet-stream;base64,{model_b64}" download="multi_model.pth">💾 Download Model</a>', unsafe_allow_html=True)
            st.markdown(f'<a href="data:application/octet-stream;base64,{preprocessor_b64}" download="multi_preprocessor.pkl">🧰 Download Preprocessor</a>', unsafe_allow_html=True)

            
            
            st.markdown("### 🛠 Cara Menggunakan Model dan Preprocessor untuk Multi-Class Classification:")
            st.code("""
            # 1. Import Library
            import torch
            import joblib
            import numpy as np

            # 2. Load Preprocessor
            pipeline = joblib.load('preprocessor.pkl')

            # 3. Transform Data Baru
            X_new = pipeline.transform(data_baru)  # data_baru = DataFrame dengan format fitur yang sama

            # 4. Load Trained Model
            model = torch.load('trained_model.pth')
            model.eval()

            # 5. Inference
            with torch.no_grad():
                input_tensor = torch.tensor(X_new, dtype=torch.float32)
                prediction = model(input_tensor)

                # 6. Ambil kelas dengan probabilitas tertinggi
                pred_class = torch.argmax(prediction, dim=1)
                print(f"Predicted class: {pred_class.item()}")
            """, language="python")

        except:
            st.error("""
                **Kesalahan input!**  
                Silakan periksa hal-hal berikut:
                - Refresh ulang halaman browser anda.
                - Pastikan data yang tidak diperlukan sudah dihapus manual atau menggunakan menu "Select Target Drop".
                - Pastikan Kolom "Select Target Column" sudah diubah manual menjadi angka.
                - Pastikan Kolom "Select Target Column' hanya berisi satu kolom saja.
                - Pastikan Kolom "Select Numerical Column" , "Select One Hot Categorical Column" , "Select Ordinal Categorical Column" diinput dengan benar.
                """)
# ---Library---
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from function import *
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Build Your Own Model - Binari Classification", 
                   page_icon="🤖",
                   layout='wide'
                   )

# ---Main---
st.markdown(
    """
    <h3 style='text-align: center;'>🧾 Build Your Own Model Classification using Neural Network</h3>
    <h5 style='text-align: center;'>Binary Classification</h5>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("**Upload Dataset CSV :**", type=["csv"])

st.markdown("""
        📌 **Catatan:**
        Sebelum upload data, pastikan bahwa kolom target sudah diubah menjadi 
        bentuk **ordinal encoding** (misalnya: 0 = No Churn, 1 = Churn).
        """)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
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
        
        all_columns = df.columns.tolist()
    
        drop = st.multiselect("**Select Target Drop**", options=all_columns)
    
    with col4:
        df = drop_cols(df, drop)
    
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
    # with col5:
        all_columns = df.columns.tolist()
        
        target = st.selectbox("Select Target Column", options=all_columns)
        
        
    
    with col6:
        feature_cols = [col for col in all_columns if col != target]
        
        # Allow users to choose features and target dynamically
        numerical_col = st.multiselect("Select Numerical Columns", options=feature_cols)
    
    with col7:
        feature_cols = [col for col in all_columns if col != target]   
         
        # Allow users to choose features and target dynamically
        cat_onehot_col = st.multiselect("Select One Hot Categorical Column", options=feature_cols)
        
    with col8:
        feature_cols = [col for col in all_columns if col != target]   
         
        # Allow users to choose features and target dynamically
        cat_ordinal_col = st.multiselect("Select Ordinal Categorical Column", options=feature_cols)
    
    # Tentukan fitur dan label
    X = df.drop(target, axis=1)
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
    preprocessor = ColumnTransformer(transformers=transformers)

    # Buat Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    
    
    # Split Data Train Test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
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
        # Create model
        # konversi ke tensor pytorch (karena format sebelumnya adalah dataframe, maka diganti terlebih dahulu ke numpy)
        X_train_tensor = torch.tensor(X_train_processed,dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_processed,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(dtype="float32")).unsqueeze(dim=1)
        y_test_tensor = torch.tensor(y_test.to_numpy(dtype="float32")).unsqueeze(dim=1)

        # X_train_tensor = torch.tensor(X_train)
        # X_test_tensor = torch.tensor(X_test)
        # y_train_tensor = torch.tensor(y_train).unsqueeze(dim=1)
        # y_test_tensor = torch.tensor(y_test).unsqueeze(dim=1)

        # X_train_tensor.shape, 
        # X_test_tensor.shape, 
        # y_train_tensor.shape, 
        # y_test_tensor.shape
        
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
    
    
    
    st.write("**Layer Setting :**")
    
    col11, col12 = st.columns(2)
    
    with col11:
        batch_size = st.number_input("Select Batch Size", value=32, key='batch_size')

    with col12:
        # Dynamic layer configuration
        num_layers = st.number_input("Number of Hidden Layers", value=1, key='num_layers')
    
    
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
        loss_choice = st.selectbox("📉 Pilih Loss Function", ["BCELoss"])
        st.markdown("""
        📌 **Catatan:**
        - `BCELoss` digunakan untuk **binary classification** (output pakai sigmoid).
        """)

    with col16:
        # Streamlit: Pilihan Optimizer
        optimizer_choice = st.selectbox("⚙️ Pilih Optimizer", ["Adam", "SGD", "RMSprop"])
        
            
    # Layers
    layers = st.write("**Layer Configuration :**")
    layers = []
    
    
    for i in range(num_layers):
        with st.expander(f"Layer {i+1} Configuration"):
            units = st.slider(f"🧠 Units in Layer {i+1}", 16, 256, 16, key=f'units_{i}')
            activation = st.selectbox(
                f"⚡ Activation Function for Layer {i+1}",
                options=['relu', 'tanh', 'selu'],
                key=f'act_{i}'
            )
            
            input_dim = X_train_tensor.shape[1]
            
            
            # Tambahkan linear layer
            if i == 0:
                layers.append(nn.Linear(input_dim, units))  # layer pertama
            else:
                prev_units = layers[-2].out_features  # ambil jumlah unit dari layer linear sebelumnya
                layers.append(nn.Linear(prev_units, units))
            
            # Tambahkan fungsi aktivasi sesuai pilihan
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'selu':
                layers.append(nn.SELU())

    # Output layer
    last_units = layers[-2].out_features if len(layers) >= 2 else units  # fallback jika cuma 1 layer
    layers.append(nn.Linear(last_units, 1))
    layers.append(nn.Sigmoid())  # untuk binary classification

    st.write("**Model Architecture :**")
    # Bangun model dengan nn.Sequential
    model = nn.Sequential(*layers)
    st.write(model)
    
    
    
    if st.button("Training & Testing Model"):
        try:
            # Train model
            with st.spinner("Running ... please wait ⏳"):
                # st.write("**Training & Testing Model (Epochs) :**")
                # Built model dynamically
                # Inisialisasi Loss Function
                if loss_choice == "BCELoss":
                    criterion = nn.BCELoss()
                elif loss_choice == "CrossEntropyLoss":
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
                
                epochs = epochs
                
                progress_placeholder = st.empty()

                for epoch in range(epochs):
                    model.train()
                    
                    output_train = model(X_train_tensor)
                    
                    loss_train = criterion(output_train, y_train_tensor)
                    
                    losses_train.append(loss_train.item())
                    
                    output_label_train = (output_train >= 0.5).float()
                    
                    accuracy_train = (output_label_train == y_train_tensor).float().mean()
                    
                    optimizer.zero_grad()
                    
                    loss_train.backward()
                    
                    optimizer.step()
                    
                    # Evaluasi akurasi
                    from sklearn.metrics import confusion_matrix, classification_report

                    model.eval()
                    with torch.no_grad():
                        output_test = model(X_test_tensor)
                        
                        loss_test = criterion(output_test,  y_test_tensor)
                        
                        losses_test.append(loss_test.item())
                        
                        output_label_test = (output_test >= 0.5).float()
                                
                        accuracy_test = (output_label_test == y_test_tensor).float().mean()
                    
                    if (epoch + 1) % 10 == 0:
                        st.write(f"Epoch : [{epoch+1}/{epochs}] | Loss_train : {loss_train.item():.4f} | Loss_test : {loss_test.item():.4f} | Accuracy_train : {accuracy_train:.4f}  | Accuracy Test : {accuracy_test:.4f}")
                    
                
                # Plot loss per epoch
                fig, ax = plt.subplots()
                ax.plot(range(epochs), losses_train, label='Train Loss')
                ax.plot(range(epochs), losses_test, label='Test Loss')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Loss per Epoch")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                                
                # konversi ke numpy untuk sklearn
                y_true_train = y_train_tensor.numpy()
                y_pred_train = output_label_train.numpy()

                # konversi ke numpy untuk sklearn
                y_true_test = y_test_tensor.numpy()
                y_pred_test = output_label_test.numpy()
                
                # Confusion matrix data train
                cm_train = confusion_matrix(y_true_train, y_pred_train)
                
                # Confusion matrix data test
                cm_test = confusion_matrix(y_true_test, y_pred_test)
                
                
                col17, col18 = st.columns(2)

                with col17:
                    st.write("**Confusion Matrix - Training Data**")
                    fig_train, ax_train = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['No Churn', 'Churn'],
                                yticklabels=['No Churn', 'Churn'],
                                ax=ax_train)
                    ax_train.set_xlabel('Predicted Label')
                    ax_train.set_ylabel('True Label')
                    ax_train.set_title('Confusion Matrix (Train)')
                    st.pyplot(fig_train)

                with col18:
                    st.write("**Confusion Matrix - Testing Data**")
                    fig_test, ax_test = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['No Churn', 'Churn'],
                                yticklabels=['No Churn', 'Churn'],
                                ax=ax_test)
                    ax_test.set_xlabel('Predicted Label')
                    ax_test.set_ylabel('True Label')
                    ax_test.set_title('Confusion Matrix (Test)')
                    st.pyplot(fig_test)


                col19, col20 = st.columns(2)

                with col19:
                    st.write("**Classification Report - Training Data**")
                    cr_train = classification_report(y_true_train, y_pred_train)
                    st.code(cr_train)  # atau bisa juga st.text(cr_train)

                with col20:
                    st.write("**Classification Report - Testing Data**")
                    cr_test = classification_report(y_true_test, y_pred_test)
                    st.code(cr_test)
                    
                    
            
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

                model_link = f'<a href="data:application/octet-stream;base64,{model_b64}" download="trained_model.pth">💾 Download Trained Model (.pth)</a>'
                preprocessor_link = f'<a href="data:application/octet-stream;base64,{preprocessor_b64}" download="preprocessor.pkl">🧰 Download Preprocessor (.pkl)</a>'

                st.markdown(model_link, unsafe_allow_html=True)
                st.markdown(preprocessor_link, unsafe_allow_html=True)
                
                
                st.markdown("### 🛠 Cara Menggunakan Model dan Preprocessor:")
                st.code("""
                # 1. Import Library
                import torch
                import joblib

                # 2. Load Preprocessor
                pipeline = joblib.load('preprocessor.pkl')

                # 3. Transform Data Baru
                X_new = pipeline.transform(data_baru)

                # 4. Load Trained Model
                model = torch.load('trained_model.pth')
                model.eval()

                # 5. Inference
                with torch.no_grad():
                    input_tensor = torch.tensor(X_new, dtype=torch.float32)
                    prediction = model(input_tensor)
                    label = (prediction >= 0.5).float()
                    print(label)
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
            
            
            
            
            
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
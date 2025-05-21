# ---Library---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- Load data ---
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


#---Preprocessing function---
def preprocessing_data(numerical_cols, cat_onehot_cols, cat_ordinal_cols):
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', MinMaxScaler(), numerical_cols),
            ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore'), cat_onehot_cols),
            ('ordinal', OrdinalEncoder(), cat_ordinal_cols)
        ],
        remainder = 'drop'
    )
    return preprocessor


def drop_cols(df, cols):
    df = df.drop(columns = cols)
    return df



def get_feature_names(preprocessor):
    output_features = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'drop':
            continue
        elif transformer == 'passthrough':
            output_features.extend(cols)
        else:
            try:
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out(cols)
                else:
                    names = cols
                output_features.extend(names)
            except:
                output_features.extend(cols)

    return output_features

# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import pickle

# Load models
@st.cache_resource
def load_model(model_name):
    model_path = f"../models/{model_name}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Page title
st.title("Credit Card Fraud Detection 🔍💳")

# Sidebar model selector
model_choice = st.sidebar.selectbox("Select Model", ["final_randomforest", "final_xgboost", "final_logisticregression"])
model = load_model(model_choice)

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Sample Input Data", data.head())

    if st.button("Predict Fraud"):
        # Basic check for model input columns (you can make this stricter)
        if len(data.columns) < 5:
            st.warning("Uploaded data seems incomplete. Please check your input file.")
        else:
            # Make predictions
            predictions = model.predict(data)
            data["Fraud_Prediction"] = predictions

            st.write("✅ Prediction Results", data.head())

            # Download predictions
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

else:
    st.info("👈 Please upload a file to begin.")

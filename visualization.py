import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch
import io


with open("model_1119.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

with open("threshold_1119.txt", "r") as f:
    threshold = float(f.read())

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#--------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_embeddings(text):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings

# --------------------
st.title("ReAdmit")

uploaded_file = st.file_uploader("Upload csv file", type=["csv"])
batch_predict_button = st.button("Predict")

with st.form("Prediction_form"):
    st.header("Manual data entry")

    def create_input_section(section_title, features, section_color):
        st.markdown(f"### <span style='color:{section_color}'>{section_title}</span>", unsafe_allow_html=True)
        cols = st.columns(4)
        inputs = {}
        for i, feature in enumerate(features):
            col_idx = i % 4
            inputs[feature] = cols[col_idx].text_input(f"{feature}", value="")
        return inputs

    general_info = ['admission_age', 'genderscore', 'los_hospital', 'los_icu']
    vital_signs = [
        'heart_rate_24hmean', 'heart_rate_24hmax', 'heart_rate_24hmin', 'heart_rate_24hfinal',
        'sbp_ni_24hmean', 'sbp_ni_24hmax', 'sbp_ni_24hmin', 'sbp_ni_24hfinal', 'dbp_ni_24hmean',
        'dbp_ni_24hmax', 'dbp_ni_24hmin', 'dbp_ni_24hfinal', 'mbp_ni_24hmax', 'mbp_ni_24hmin',
        'mbp_ni_24hfinal', 'spo2_24hmean', 'spo2_24hmax', 'spo2_24hmin', 'spo2_24hfinal',
        'temperature_24hfinal'
    ]
    other_signs = ['urineoutput_24hr', 'ras', 'gcs', 'shock_index', 'charlson', 'mechanical_ventilation_time',
        'invasive_ventilation']
    comorbidities = [
        'hypertension', 'diabetes_without_complications', 'diabetes_with_complications', 'hematologic_malignancies',
        'metastatic_cancer', 'anemia', 'coagulation_disorders', 'pulmonary_circulatory_diseases',
        'chronic_kidney_disease', 'arrhythmias', 'heart_failure', 'myocardial_infarction',
        'liver_disorders', 'stroke_and_cerebrovascular_disease', 'gastrointestinal_bleeding',
        'paralysis', 'peptic_ulcer', 'electrolyte_imbalance', 'substance_abuse', 'psychosis', 'sepsis',
        'overweight_and_obesity'
    ]
    lab_results = [
        'wbc', 'aniongap', 'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine', 'alt', 'ast',
        'bilirubin_total', 'glucose', 'sodium', 'potassium', 'inr', 'pt', 'ptt', 'hematocrit',
        'hemoglobin', 'albumin', 'mch', 'platelet', 'rbc', 'rdw', 'phosphate', 'mg', 'lactate',
        'bun_creatinine', 'ph', 'be', 'pao2', 'paco2', 'o2_flow'
    ]


    general_info_inputs = create_input_section("general information", general_info, "blue")
    vital_signs_inputs = create_input_section("vital signs", vital_signs, "green")
    other_signs_inputs = create_input_section("other signs", other_signs, "orange")
    comorbidities_inputs = create_input_section("comorbidities", comorbidities, "red")
    lab_results_inputs = create_input_section("lab results", lab_results, "purple")

    st.header("Text input")
    text_input = st.text_area("The text of the most recent ICU radiology report", "")

    submitted = st.form_submit_button("Pridict")

if batch_predict_button and uploaded_file:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("Preview of uploaded file")
        st.write(uploaded_data.head())

        struct_features = general_info + vital_signs + other_signs + comorbidities + lab_results
        text_column = 'radiology_text'  

        missing_columns = [col for col in struct_features + [text_column] if col not in uploaded_data.columns]
        if missing_columns:
            st.error(f"Missing the following columns：{', '.join(missing_columns)}")
        else:
            st.success("All necessary columns have been detected, start prediction.")

            predictions = []

            for _, row in uploaded_data.iterrows():
                try:
                    struct_input = row[struct_features].values.reshape(1, -1)

                    text_input = row[text_column]
                    embeddings = generate_embeddings(text_input)
                    embeddings_reduced = pca.transform([embeddings])  

                    final_input = np.hstack([struct_input, embeddings_reduced])

                    prob = model.predict_proba(final_input)[:, 1][0]
                    result = "Hign risk" if prob >= threshold else "Low risk"

                    predictions.append({
                        "hadm_id": row.get("hadm_id", None), 
                        "Risk Probability": prob,
                        "Prediction results": result
                    })
                except Exception as e:
                    st.warning(f"Error processing row data：{e}")
                    predictions.append({
                        "hadm_id": row.get("hadm_id", None),
                        "Risk Probability": None,
                        "Prediction results": "Error"
                    })

            predictions_df = pd.DataFrame(predictions)
            st.write(predictions_df)

            csv_buffer = io.StringIO()
            predictions_df.to_csv(csv_buffer, index=False)
            csv_download = csv_buffer.getvalue()

            st.download_button(
                label="Download prediction results",
                data=csv_download,
                file_name="Prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error：{e}")

#------------------
if submitted:
    try:
        embeddings = generate_embeddings(text_input)
        embeddings_reduced = pca.transform([embeddings])

        struct_input = {
            **general_info_inputs, **vital_signs_inputs, **other_signs_inputs, **comorbidities_inputs,
            **lab_results_inputs
        }
        struct_df = pd.DataFrame([struct_input])
        final_input = np.hstack([struct_df.values, embeddings_reduced])

        prediction = model.predict_proba(final_input)[:, 1]
        result = "High risk" if prediction[0] >= threshold else "Low risk"

        # 显示结果
        st.subheader(f"Prediction_results：{result}")
    except Exception as e:
        st.error(f"Error：{e}")

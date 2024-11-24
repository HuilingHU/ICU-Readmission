test_data = pd.read_sql_query('SELECT * FROM mimiciv_new.male', engine)
# Use subgroup files separately: female, more65, less65,ccu, tsicu, msicu, sicu, micu, cvicu
selected_features = ['admission_age', 'genderscore', 'los_hospital', 'los_icu', 
                     'heart_rate_24hmean', 'heart_rate_24hmax', 'heart_rate_24hmin',
                     'heart_rate_24hfinal','sbp_ni_24hmean', 'sbp_ni_24hmax',
                     'sbp_ni_24hmin', 'sbp_ni_24hfinal', 'dbp_ni_24hmean',
                     'dbp_ni_24hmax', 'dbp_ni_24hmin', 'dbp_ni_24hfinal', 'mbp_ni_24hmax',
                     'mbp_ni_24hmin','mbp_ni_24hfinal', 'spo2_24hmean', 'spo2_24hmax',
                     'spo2_24hmin', 'spo2_24hfinal','temperature_24hfinal', 'urineoutput_24hr',
                     'ras', 'gcs', 'shock_index', 'charlson', 'mechanical_ventilation_time',
                     'invasive_ventilation', 'hypertension', 'diabetes_without_complications',
                     'diabetes_with_complications', 'hematologic_malignancies','metastatic_cancer',
                     'anemia', 'coagulation_disorders', 'pulmonary_circulatory_diseases',
                     'chronic_kidney_disease', 'arrhythmias', 'heart_failure', 'myocardial_infarction',
                     'liver_disorders', 'stroke_and_cerebrovascular_disease', 'gastrointestinal_bleeding',
                     'paralysis', 'peptic_ulcer', 'electrolyte_imbalance', 'substance_abuse',
                     'psychosis', 'sepsis','overweight_and_obesity', 'wbc', 'aniongap', 'bicarbonate',
                     'bun', 'calcium', 'chloride', 'creatinine', 'alt', 'ast', 'bilirubin_total', 'glucose',
                     'sodium', 'potassium', 'inr', 'pt', 'ptt', 'hematocrit','hemoglobin', 'albumin',
                     'mch', 'platelet', 'rbc', 'rdw', 'phosphate', 'mg', 'lactate','bun_creatinine',
                     'ph', 'be', 'pao2', 'paco2', 'o2_flow']
pca_features = [f'pca_{i}' for i in range(1, 101)]  

X_test = pd.concat([test_data[selected_features], test_data[pca_features]], axis=1)
y_test = test_data['positive_outcome']

loaded_model = joblib.load("model_1119.pkl")
print("Model loaded successfully.")

with open("threshold_1119.txt", "r") as f:
    loaded_threshold = float(f.read())
print(f"Threshold loaded: {loaded_threshold}")

y_test_prob = loaded_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= loaded_threshold).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auroc = roc_auc_score(y_test, y_test_prob)

print(f"Test Set Precision: {test_precision:.4f}")
print(f"Test Set Recall: {test_recall:.4f}")
print(f"Test Set F1 Score: {test_f1:.4f}")
print(f"Test Set AUROC: {test_auroc:.4f}")

N = len(test_data)                     
O = y_test.sum()                      
E = y_test_prob.sum()                  
SMR = O / E if E > 0 else np.nan       


if O > 0:
    CI_lower = SMR * np.exp(-1.96 * np.sqrt(1 / O))  
    CI_upper = SMR * np.exp(1.96 * np.sqrt(1 / O))   
else:
    CI_lower = np.nan
    CI_upper = np.nan


print(f"Group: Male")
print(f"Sample Size (N): {N}")
print(f"Observed Events (O): {O}")
print(f"Expected Events (E): {E:.2f}")
print(f"SMR: {SMR:.4f}")
print(f"95% CI: ({CI_lower:.4f}, {CI_upper:.4f})")

#forest plot
import pandas as pd

data = {
    "Group": ["total", "male", "female", "≥65", "<65", "CCU", "TSICU", "MICU/SICU", "SICU", "MICU", "CVICU"],
    "N": [9215, 5412, 3803, 4984, 4231, 956, 1143, 1450, 1478, 1706, 2271],
    "O": [551, 337, 214, 344, 207.58, 73, 85, 88, 89, 112, 88],
    "E": [499.65, 291.15, 208.50, 299.07, 200.58, 52.03, 64.07, 92.13, 83.23, 104.97, 90.85],
    "SMR": [1.103, 1.158, 1.026, 1.150, 1.032, 1.403, 1.327, 0.955, 1.069, 1.067, 0.969],
    "CI_lower": [1.014, 1.040, 0.898, 1.035, 0.901, 1.116, 1.073, 0.775, 0.869, 0.887, 0.786],
    "CI_upper": [1.199, 1.288, 1.174, 1.278, 1.183, 1.765, 1.641, 1.177, 1.316, 1.284, 1.194],
}

df = pd.DataFrame(data)

import matplotlib.pyplot as plt

df = df.iloc[::-1].reset_index(drop=True)

groups = df["Group"]
smr = df["SMR"]
ci_lower = df["CI_lower"]
ci_upper = df["CI_upper"]


plt.figure(figsize=(8, 6))
plt.errorbar(smr, groups, xerr=[smr - ci_lower, ci_upper - smr], fmt='o', color='black', ecolor='gray', capsize=3, label='SMR (95% CI)')
plt.axvline(x=1, color='burlywood', linestyle='--', label='SMR = 1')


plt.xlabel("SMR")
plt.title("Forest Plot for SMR by Group")
plt.legend(loc='upper right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()






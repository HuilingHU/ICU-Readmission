import pandas as pd
from sklearn.model_selection import train_test_split


import psycopg2
from sqlalchemy import create_engine


engine = create_engine('postgresql://postgres:XXXXX@localhost:5433/mimiciv2.2')


df = pd.read_sql('SELECT * FROM mimiciv_new.prediction', engine)

train_data, temp_data = train_test_split(df, test_size=0.3, stratify=df['positive_outcome'])  
calib_data, val_data = train_test_split(temp_data, test_size=2/3, stratify=temp_data['positive_outcome'])  

train_data.to_sql('1_train_data', engine, if_exists='replace', index=False)
calib_data.to_sql('1_calib_data', engine, if_exists='replace', index=False)
val_data.to_sql('1_val_data', engine, if_exists='replace', index=False)

#IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()

imputer = IterativeImputer(estimator=lgbm, max_iter=10, random_state=42)

train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)

calib_data_imputed = pd.DataFrame(imputer.transform(calib_data), columns=calib_data.columns)
val_data_imputed = pd.DataFrame(imputer.transform(val_data), columns=val_data.columns)

train_data_imputed.to_sql('1_train_data_imputed', engine, if_exists='replace', index=False)
calib_data_imputed.to_sql('1_calib_data_imputed', engine, if_exists='replace', index=False)
val_data_imputed.to_sql('1_val_data_imputed', engine, if_exists='replace', index=False)

# Standardized
from sklearn.preprocessing import StandardScaler

exclude_columns = ['positive_outcome', 'hadm_id','pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10', 'pca_11', 'pca_12',
                   'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20', 'pca_21', 'pca_22',
                   'pca_23', 'pca_24', 'pca_25', 'pca_26', 'pca_27', 'pca_28', 'pca_29', 'pca_30', 'pca_31', 'pca_32',
                   'pca_33', 'pca_34', 'pca_35', 'pca_36', 'pca_37', 'pca_38', 'pca_39', 'pca_40', 'pca_41', 'pca_42',
                   'pca_43', 'pca_44', 'pca_45', 'pca_46', 'pca_47', 'pca_48', 'pca_49', 'pca_50', 'pca_51', 'pca_52', 'pca_53', 'pca_54', 'pca_55', 'pca_56', 'pca_57', 'pca_58', 'pca_59', 'pca_60', 'pca_61', 'pca_62',
                       'pca_63', 'pca_64', 'pca_65', 'pca_66', 'pca_67', 'pca_68', 'pca_69', 'pca_70', 'pca_71', 'pca_72',
                       'pca_73', 'pca_74', 'pca_75', 'pca_76', 'pca_77', 'pca_78', 'pca_79', 'pca_80', 'pca_81', 'pca_82',
                       'pca_83', 'pca_84', 'pca_85', 'pca_86', 'pca_87', 'pca_88', 'pca_89', 'pca_90', 'pca_91', 'pca_92',
                       'pca_93', 'pca_94', 'pca_95', 'pca_96', 'pca_97', 'pca_98', 'pca_99', 'pca_100']

columns_to_standardize = [col for col in train_data_imputed.columns if col not in exclude_columns and train_data_imputed[col].nunique() > 2]

scaler = StandardScaler()

train_data_standardized = train_data_imputed.copy()
train_data_standardized[columns_to_standardize] = scaler.fit_transform(train_data_imputed[columns_to_standardize])

train_data_standardized.to_sql('1_train_data_standardized', engine, if_exists='replace', index=False)


#SMOTE
from imblearn.over_sampling import SMOTE

X_train = train_data_standardized.drop(['positive_outcome', 'hadm_id', 'racescore', 'careunitscore'], axis=1)
y_train = train_data_standardized['positive_outcome']

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
train_data_resampled = X_train_resampled.copy()
train_data_resampled['positive_outcome'] = y_train_resampled
train_data_resampled.to_sql('1_train_data_resampled', engine, if_exists='replace', index=False)

# feature selection(Choose one method each time)
#1.LASSO
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

exclude_pca_columns = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10', 'pca_11', 'pca_12',
                       'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20', 'pca_21', 'pca_22',
                       'pca_23', 'pca_24', 'pca_25', 'pca_26', 'pca_27', 'pca_28', 'pca_29', 'pca_30', 'pca_31', 'pca_32',
                       'pca_33', 'pca_34', 'pca_35', 'pca_36', 'pca_37', 'pca_38', 'pca_39', 'pca_40', 'pca_41', 'pca_42',
                       'pca_43', 'pca_44', 'pca_45', 'pca_46', 'pca_47', 'pca_48', 'pca_49', 'pca_50', 'pca_51', 'pca_52', 'pca_53', 'pca_54', 'pca_55', 'pca_56', 'pca_57', 'pca_58', 'pca_59', 'pca_60', 'pca_61', 'pca_62',
                       'pca_63', 'pca_64', 'pca_65', 'pca_66', 'pca_67', 'pca_68', 'pca_69', 'pca_70', 'pca_71', 'pca_72',
                       'pca_73', 'pca_74', 'pca_75', 'pca_76', 'pca_77', 'pca_78', 'pca_79', 'pca_80', 'pca_81', 'pca_82',
                       'pca_83', 'pca_84', 'pca_85', 'pca_86', 'pca_87', 'pca_88', 'pca_89', 'pca_90', 'pca_91', 'pca_92',
                       'pca_93', 'pca_94', 'pca_95', 'pca_96', 'pca_97', 'pca_98', 'pca_99', 'pca_100']

X_train_filtered = X_train_resampled.drop(columns=exclude_pca_columns)

lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso.fit(X_train_filtered, y_train_resampled)

selected_features = X_train_filtered.columns[lasso.coef_ != 0]
#2.Random Forest
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

exclude_pca_columns = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10', 'pca_11', 'pca_12',
                       'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pca_17', 'pca_18', 'pca_19', 'pca_20', 'pca_21', 'pca_22',
                       'pca_23', 'pca_24', 'pca_25', 'pca_26', 'pca_27', 'pca_28', 'pca_29', 'pca_30', 'pca_31', 'pca_32',
                       'pca_33', 'pca_34', 'pca_35', 'pca_36', 'pca_37', 'pca_38', 'pca_39', 'pca_40', 'pca_41', 'pca_42',
                       'pca_43', 'pca_44', 'pca_45', 'pca_46', 'pca_47', 'pca_48', 'pca_49', 'pca_50', 'pca_51', 'pca_52', 'pca_53', 'pca_54', 'pca_55', 'pca_56', 'pca_57', 'pca_58', 'pca_59', 'pca_60', 'pca_61', 'pca_62',
                       'pca_63', 'pca_64', 'pca_65', 'pca_66', 'pca_67', 'pca_68', 'pca_69', 'pca_70', 'pca_71', 'pca_72',
                       'pca_73', 'pca_74', 'pca_75', 'pca_76', 'pca_77', 'pca_78', 'pca_79', 'pca_80', 'pca_81', 'pca_82',
                       'pca_83', 'pca_84', 'pca_85', 'pca_86', 'pca_87', 'pca_88', 'pca_89', 'pca_90', 'pca_91', 'pca_92',
                       'pca_93', 'pca_94', 'pca_95', 'pca_96', 'pca_97', 'pca_98', 'pca_99', 'pca_100']

X_train_filtered = X_train_resampled.drop(columns=exclude_pca_columns)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_filtered, y_train_resampled)

perm_importance = permutation_importance(rf, X_train_filtered, y_train_resampled, n_repeats=10, random_state=42)

importances = perm_importance.importances_mean
std = perm_importance.importances_std

important_features = X_train_filtered.columns[importances > 0]


#prediction
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

def get_best_threshold_and_metrics(y_true, y_pred_prob, method=1):
    if method == 1:  
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        f1_scores = 2 * recall * precision / (recall + precision)
        idx_best_f1 = np.argmax(f1_scores)
        threshold = thresholds[idx_best_f1]
        
    elif method == 2:  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        specificities = 1 - fpr
        idx_best = np.argmin(np.abs(specificities - tpr))
        threshold = thresholds[idx_best]
    
    elif method == 3: 
        value = 0.9
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        closest_index = np.argmin(np.abs(tpr - value))
        threshold = thresholds[closest_index]

    y_pred = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = 2 * sensitivity * precision / (sensitivity + precision)

    print(f'Best Threshold: {threshold:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Sensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (PPV): {precision:.4f}')
    print(f'NPV: {npv:.4f}')
    
    return threshold


train_data = pd.read_sql_query('SELECT * FROM "public"."1_train_data_imputed"', engine)
calib_data = pd.read_sql_query('SELECT * FROM "public"."1_calib_data_imputed"', engine)
val_data = pd.read_sql_query('SELECT * FROM "public"."1_val_data_imputed"', engine)

pca_features = [f'pca_{i}' for i in range(1, 101)] 

X_train = pd.concat([train_data[selected_features], train_data[pca_features]], axis=1)
#If the random forest method is used, selected_features should be replaced with important_features
#If only structured data is used, only train_data[selected_features] is kept here
y_train = train_data['positive_outcome']

X_calib = pd.concat([calib_data[selected_features], calib_data[pca_features]], axis=1)
y_calib = calib_data['positive_outcome']

X_val = pd.concat([val_data[selected_features], val_data[pca_features]], axis=1)
y_val = val_data['positive_outcome']

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

search_space = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3, 'uniform'),
    'subsample': (0.5, 1.0, 'uniform'),
    'colsample_bytree': (0.5, 1.0, 'uniform'),
    'scale_pos_weight': (1, pos_weight * 2, 'uniform') 
}

opt = BayesSearchCV(
    model,
    search_space,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
opt.fit(X_train, y_train)

best_model = opt.best_estimator_

calibrated_model = CalibratedClassifierCV(estimator=best_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

y_calib_prob_calibrated = calibrated_model.predict_proba(X_calib)[:, 1]
y_val_prob_calibrated = calibrated_model.predict_proba(X_val)[:, 1]

threshold = get_best_threshold_and_metrics(y_calib, y_calib_prob_calibrated, method=1)
y_val_pred = (y_val_prob_calibrated >= threshold).astype(int)

val_roc_auc = roc_auc_score(y_val, y_val_prob_calibrated)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_brier = brier_score_loss(y_val, y_val_prob_calibrated)

print(f'Validation Set - AUROC: {val_roc_auc:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
      f'Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, Brier Score: {val_brier:.4f}')

fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_prob_calibrated, n_bins=30)

plt.figure(figsize=(10, 8))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='green', label="Isotonic Calibration") 
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.title("Calibration Curve - Isotonic Regression")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend()
plt.show()

fpr, tpr, thresholds = roc_curve(y_val, y_val_prob_calibrated)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='green', label='Calibrated Model (AUROC = {:.4f})'.format(val_roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUROC = 0.5)')
plt.title('ROC Curve - Isotonic Regression Calibration')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#SHAP
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val)

plt.figure()
shap.summary_plot(shap_values, X_val, plot_type="bar")

#save model
import joblib

model_path = "model_XX.pkl"
threshold_path = "threshold_XX.txt"

joblib.dump(calibrated_model, model_path)
print(f"Model saved to {model_path}")

with open(threshold_path, "w") as f:
    f.write(str(threshold))
print(f"Threshold saved to {threshold_path}")


import psycopg2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sqlalchemy import create_engine

# Step 1
engine = create_engine('postgresql://postgres:XXX@localhost:5433/mimiciv2.2')

train_df = pd.read_sql_query('SELECT * FROM "public"."train_data_resampled"', engine)
calib_df = pd.read_sql_query('SELECT * FROM "public"."calib_data_standardized"', engine)
val_df = pd.read_sql_query('SELECT * FROM "public"."val_data_standardized"', engine)

y_train = train_df['positive_outcome']
X_train_structured = train_df.drop(columns=['positive_outcome', *[f'pca_{i}' for i in range(1, 101)]]).to_numpy()
X_train_text = train_df[[f'pca_{i}' for i in range(1, 101)]].to_numpy()

y_val = val_df['positive_outcome']
X_val_structured = val_df.drop(columns=['positive_outcome', *[f'pca_{i}' for i in range(1, 101)]]).to_numpy()
X_val_text = val_df[[f'pca_{i}' for i in range(1, 101)]].to_numpy()

y_calib = calib_df['positive_outcome']
X_calib_structured = calib_df.drop(columns=['positive_outcome', *[f'pca_{i}' for i in range(1, 101)]]).to_numpy()
X_calib_text = calib_df[[f'pca_{i}' for i in range(1, 101)]].to_numpy()

# Step 2
structured_input = Input(shape=(X_train_structured.shape[1],), name="structured_data")
x_structured = Dense(64, activation='relu')(structured_input)
x_structured = Dense(32, activation='relu')(x_structured)

text_input = Input(shape=(100,), name="text_data")
x_text = tf.keras.layers.Reshape((100, 1))(text_input)
x_text = Conv1D(64, 3, activation='relu')(x_text)
x_text = GlobalMaxPooling1D()(x_text)
x_text = Dense(32, activation='relu')(x_text)

combined = Concatenate()([x_structured, x_text])
x = Dense(64, activation='relu')(combined)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid', name="output")(x)

model = Model(inputs=[structured_input, text_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')])

#step 3 
X_train_structured = X_train_structured.to_numpy() if hasattr(X_train_structured, 'to_numpy') else X_train_structured
X_train_text = X_train_text.to_numpy() if hasattr(X_train_text, 'to_numpy') else X_train_text
X_val_structured = X_val_structured.to_numpy() if hasattr(X_val_structured, 'to_numpy') else X_val_structured
X_val_text = X_val_text.to_numpy() if hasattr(X_val_text, 'to_numpy') else X_val_text

print(f"X_train_structured shape: {X_train_structured.shape}")
print(f"X_train_text shape: {X_train_text.shape}")
print(f"X_val_structured shape: {X_val_structured.shape}")
print(f"X_val_text shape: {X_val_text.shape}")

history = model.fit(
    {"structured_data": X_train_structured, "text_data": X_train_text},
    y_train,
    validation_data=({"structured_data": X_val_structured, "text_data": X_val_text}, y_val),
    epochs=20,
    batch_size=32
)


# Step 4
from sklearn.metrics import f1_score, roc_curve

X_calib_structured = X_calib_structured.to_numpy() if hasattr(X_calib_structured, 'to_numpy') else X_calib_structured
X_calib_text = X_calib_text.to_numpy() if hasattr(X_calib_text, 'to_numpy') else X_calib_text

print(f"X_calib_structured shape: {X_calib_structured.shape}")
print(f"X_calib_text shape: {X_calib_text.shape}")

y_calib_pred_prob = model.predict({"structured_data": X_calib_structured, "text_data": X_calib_text}).flatten()

fpr, tpr, thresholds = roc_curve(y_calib, y_calib_pred_prob)
best_threshold = 0.5
best_f1_score = 0

for threshold in thresholds:
    y_calib_pred = (y_calib_pred_prob >= threshold).astype(int)
    f1 = f1_score(y_calib, y_calib_pred)
    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

print(f"Best threshold based on F1 Score: {best_threshold}")

# Step 5
y_calib_pred_best = (y_calib_pred_prob >= best_threshold).astype(int)

accuracy = np.mean(y_calib_pred_best == y_calib)
precision = precision_score(y_calib, y_calib_pred_best)
recall = recall_score(y_calib, y_calib_pred_best)
f1 = f1_score(y_calib, y_calib_pred_best)
brier = brier_score_loss(y_calib, y_calib_pred_prob)
auroc = roc_auc_score(y_calib, y_calib_pred_prob)

print("Optimal Threshold Metrics (Calibration Set):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Sensitivity (Recall): {recall}")
print(f"F1 Score: {f1}")
print(f"Brier Score: {brier}")
print(f"AUROC: {auroc}")

# Step 6
fpr, tpr, _ = roc_curve(y_calib, y_calib_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color="red", label=f"ROC Curve (AUROC = {auroc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Calibration Set)")
plt.legend(loc="lower right")
plt.show()

prob_true, prob_pred = calibration_curve(y_calib, y_calib_pred_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', color="red", label="Calibration Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Calibration Set)")
plt.legend(loc="upper left")
plt.show()

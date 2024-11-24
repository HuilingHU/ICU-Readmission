import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import ast
import joblib

engine = create_engine('postgresql://postgres:XXX@localhost:5433/mimiciv2.2')

cleaned_text_df = pd.read_sql(
    "SELECT hadm_id, cleaned_text FROM mimiciv_new.text", engine
)
embeddings_df = pd.read_sql(
    "SELECT hadm_id, embeddings FROM mimiciv_new.embeddings", engine
)
pca_data_df = pd.read_sql(
    "SELECT * FROM mimiciv_new.pca", engine
)


embeddings_df["embeddings"] = embeddings_df["embeddings"].apply(
    lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
)

pca_model = joblib.load('pca_model.pkl')

loadings = pca_model.components_

component_names = ["PCA_82", "PCA_38", "PCA_61", "PCA_52", "PCA_59"]
important_components = [81, 37, 60, 51, 58] 

results = {}
for component_index in important_components:
    important_loading = loadings[component_index]
    most_contributing_features = np.argsort(np.abs(important_loading))[::-1]
    top_features = most_contributing_features[:10] 
    results[f"PCA_{component_index + 1}"] = {
        "top_features": top_features,
        "loadings": important_loading[top_features],
    }

def extract_embeddings(row, top_features):
    embeddings = np.array(row["embeddings"])
    return embeddings[top_features] if embeddings.ndim > 0 else []

for component_name, component_index in zip(component_names, important_components):
    top_features = results[f"PCA_{component_index + 1}"]["top_features"]
    embeddings_df[component_name] = embeddings_df.apply(
        lambda row: extract_embeddings(row, top_features), axis=1
    )

merged_df = pd.merge(cleaned_text_df, embeddings_df, on='hadm_id')

thresholds = {}
quantile = 0.5 

for component_name in component_names:
    merged_df[f"{component_name}_sum"] = merged_df[component_name].apply(
        lambda x: np.sum(np.abs(x)) if len(x) > 0 else 0
    )
    thresholds[component_name] = merged_df[f"{component_name}_sum"].quantile(quantile)

print("Thresholds for each PCA component:", thresholds)

def filter_relevant_text_all(row, thresholds):
    for component_name, threshold in thresholds.items():
        embeddings = row[component_name]
        if len(embeddings) == 0 or np.sum(np.abs(embeddings)) <= threshold:
            return False
    return True

relevant_rows_all = merged_df[
    merged_df.apply(lambda row: filter_relevant_text_all(row, thresholds), axis=1)
]

output_file_csv = "relevant_texts_all_components_1.csv"
relevant_data = []

for index, row in relevant_rows_all.iterrows():
    row_data = {
        "HADM_ID": row["hadm_id"],
        "Cleaned Text": row["cleaned_text"]
    }
    for component_name in component_names:
        row_data[f"{component_name} Embeddings"] = row[component_name]
    relevant_data.append(row_data)

relevant_df = pd.DataFrame(relevant_data)
relevant_df.to_csv(output_file_csv, index=False, encoding="utf-8")

print(f"Relevant texts for all components have been saved to {output_file_csv}")

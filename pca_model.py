import pandas as pd
import ast
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
import pickle 

engine = create_engine('postgresql://postgres:XXX@localhost:5433/mimiciv2.2')

data = pd.read_sql('SELECT * FROM mimiciv_new.radiology_embeddings_total', engine)

def parse_embeddings(x):
    if pd.isnull(x) or x == '':
        return [] 
    try:
        return ast.literal_eval(x)  
    except (ValueError, SyntaxError):
        return []

embeddings = data['embeddings'].apply(parse_embeddings).apply(pd.Series)  
embeddings.columns = [f'embedding_{i+1}' for i in range(768)]

pca = PCA(n_components=100)
embeddings_reduced = pca.fit_transform(embeddings.fillna(0)) 
embeddings_reduced = pd.DataFrame(embeddings_reduced, columns=[f'pca_{i+1}' for i in range(100)])

with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("save as pca_model.pkl")

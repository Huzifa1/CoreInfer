
import pickle

cluster_path = "./cluster/Llama3-8B_QA/"
with open(f'{cluster_path}/cluster_activation/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open(f'{cluster_path}/cluster_activation/mlb_model.pkl', 'rb') as file:
    mlb_loaded = pickle.load(file)
        
print(mlb_loaded)
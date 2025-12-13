import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

def compute_pca(encodings, pca_save_path,n_components=2):
    # """
    # encodings: list/array where each row is [ID, f1, f2, ..., fN]
    # n_components: number of PCA dimensions
    # """
    # encodings = np.array(encodings)
    
    # ids = encodings[:, 0]                    # first column
    # data = encodings[:, 1:].astype(float)    # remaining columns as float

    # # Fit PCA
    # pca_model = PCA(n_components=n_components)
    # transformed = pca_model.fit_transform(data)

    # # Reattach IDs
    # output = np.column_stack([ids, transformed])

    # # return output, pca_model

    X = np.stack(encodings,axis=0)
    X = np.unique(X, axis=0)
    print('pca.py X',X.shape)
    pca_model = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))
    pca_model.fit(X)
    # X_embedded = pca_model.transform(X)
    # X_embedded.shape
    with open(pca_save_path, 'wb') as f:
        pickle.dump(pca_model, f)
    return pca_model

#         # pca_x,pca_y = [],[]
#     pca_embed_dict = dict()
#     for i in tqdm(range(2000000)):
#         file = esm_files[i]
#         pep = file.split('/')[-1].split('.')[0]
#         pca_embed_dict[pep] = pca_model.transform(encodings[i].reshape(1, -1))
    
# pca_embed_dict


def apply_pca(pca_model, encodings):
    encodings = np.array(encodings)
    transformed = pca_model.transform(encodings)
    return transformed

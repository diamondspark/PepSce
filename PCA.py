import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import os

# Calculate PCA on Btrain-ESM and apply to (D-Btrain)ESM

# NBtrain = 50000
# with open('/home/mkpandey/MyProjects/Generative_Modeling/ACP/PepSce/data/peptides_df_ESM.pkl','rb') as f:
#     df = pickle.load(f)

def pca(df, NBtrain, path):
    df_train = df.sample(n=NBtrain, random_state=42)

    X = np.stack(df_train.ESM.to_list(),axis=0)
    X_all = np.stack(df.ESM.to_list(), axis=0)
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=42))
    print(f'PCA fitting on {len(X)} peptides....')
    pca.fit(X)

    base_dir = os.path.dirname(path)
    # Change the directory to 'models' and the filename to 'pca.pkl'
    new_dir = os.path.join(base_dir, '../models')
    new_path = os.path.abspath(os.path.join(new_dir, 'pca.pkl'))
    with open(new_path,'wb') as f:
        pickle.dump(pca,f)

    print(f'PCA transformation on {len(df)} peptides....')
    X_embedded = pca.transform(X_all)
    df['pca_X']=X_embedded[:,0]
    df['pca_Y']=X_embedded[:,1] 
    df.pca_X = pd.to_numeric(df.pca_X, downcast='float')
    df.pca_Y = pd.to_numeric(df.pca_Y, downcast='float')
            
    df['pca_X16'] = df.pca_X.astype(np.float16)
    df['pca_Y16'] = df.pca_Y.astype(np.float16)
    return df

# pca(df,50000, '/home/mkpandey/MyProjects/Generative_Modeling/ACP/PepSce/data/peptides.txt')
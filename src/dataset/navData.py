import pandas as pd
import pickle
import numpy as np


# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/src')
from dataset.pca import compute_pca, apply_pca
import glob
import pickle
from tqdm import tqdm
import torch


def quantize(x, resolution= 1e-3):
    return float(np.round(x / resolution) * resolution)


class NavData():
    def __init__(self,data_path, pca_load_path=None, pca_save_path=None, precomputed_nav_data=None):
        if precomputed_nav_data:
            with open(precomputed_nav_data,'rb') as f:
                self.nav_2d, self.pep_coord_reward_dict = pickle.load(f)
        else:
            files = glob.glob(data_path+'/*.pt')
            pca_pep_esm_list = self.load_esm_means_with_pca_list(files)
            idlist, peplist, esmlist = zip(*pca_pep_esm_list)
            peplist, esmlist = list(peplist), list(esmlist)
            print('idlist, peplist, esmlist', len(idlist), len(peplist), len(esmlist))
            if pca_load_path:
                with open(pca_load_path,'rb') as f:
                    self.pca_model = pickle.load(f)
            else:
                self.pca_model = compute_pca(esmlist,pca_save_path=pca_save_path)
            self.nav_2d = apply_pca(self.pca_model, encodings=esmlist)
            self.pep_coord_reward_dict = {
                (np.float16(x), np.float16(y)): [pep, None]
                for (x, y), pep in zip(self.nav_2d, peplist)
            }
        print('nav data.py len pep_coord_reward_dict meaning how many overlaps due to downcasting in float', len(self.pep_coord_reward_dict))
        
    def load_esm_means_with_pca_list(self, esm_batch_files, max_files=None, repr_layer=6):
        """
        Loads ESM mean embeddings from batched .pt files.

        Parameters
        ----------
        esm_batch_files : list[str]
            List of batched ESM .pt files.
        max_files : int or None
            Limit number of processed batch files.
        repr_layer : int
            Layer index for mean representations.

        Returns
        -------
        pca_pep_esm_list : list[np.ndarray]
            Loaded embeddings.
        file_load_failed : list[str]
            Files that failed to load.
        """

        pca_pep_esm_list = []
        file_load_failed = []

        n = len(esm_batch_files) if max_files is None else min(max_files, len(esm_batch_files))

        for b in tqdm(range(n), desc="Loading batched ESM embeddings"):
            batch_file = esm_batch_files[b]

            try:
                batch = torch.load(batch_file)

                # batch is a dict: {label: {"mean": {layer: vector}, ...}, ...}
                for label, entry in batch.items():

                    if "mean_representations" in entry:
                        vec = entry["mean_representations"][repr_layer].numpy()

                    elif "mean" in entry:   # if you used "mean" key in new batching format
                        vec = entry["mean"][repr_layer]

                    else:
                        raise KeyError(f"No mean representation found for {label}")

                    pca_pep_esm_list.append((label,entry["peptide"],vec))

            except Exception as e:
                print(f"Error loading batch file {batch_file}: {e}")
                file_load_failed.append(batch_file)
                continue

        return pca_pep_esm_list






class PeptideDataframe():
    def __init__(self,pca_path=None,pdb_helices_path=None,precomputed_rewards=True):
        # self.df = df
        #Read peptide PCA
        with open('./../../../ACP_AMP_project/data/desc_pca_X_embedded_y_all.pkl','rb') as f:
            self.X_embedded,self.y_all = pickle.load(f)
            
        if precomputed_rewards:
            #read dataframe with 1_4M_12PDBpeptides and their descriptors
            with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/Full_Expt_w_few_inferences/1_4M_12PDBpeptides_w_esm_PCA_w_rewards.pkl','rb') as f:
                self.df = pickle.load(f)
            
        else:
            #read dataframe with 440K peptides and their descriptors 
            with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/data/443K_alpha_helix_w_ifeat_modlamp_desc.pkl','rb') as f:
                self.df = pickle.load(f)

            self.df['pca_X']=self.X_embedded[:,0][np.where(self.y_all==0)]
            self.df['pca_Y']=self.X_embedded[:,1][np.where(self.y_all==0)]    
            self.df.pca_X = pd.to_numeric(self.df.pca_X, downcast='float')
            self.df.pca_Y = pd.to_numeric(self.df.pca_Y, downcast='float')

            self.df['pca_X16'] = self.df.pca_X.astype(np.float16)
            self.df['pca_Y16'] = self.df.pca_Y.astype(np.float16)
        self.pep_coord_reward_dict = dict()
        for i in (range(len(self.df))):
            key = (self.df.iat[i,6],self.df.iat[i,7])
            value = [self.df.iat[i,2],self.df.iat[i,4]]
            self.pep_coord_reward_dict[key]=value
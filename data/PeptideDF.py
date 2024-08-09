import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np

class PeptideDataframe():
    def __init__(self, precomp_rew_path ):#   pca_path=None,pdb_helices_path=None,precomputed_rewards=True):
        # self.df = df
        # #Read peptide PCA
        # with open('./../../../ACP_AMP_project/data/desc_pca_X_embedded_y_all.pkl','rb') as f:
        #     self.X_embedded,self.y_all = pickle.load(f)
            
        if precomp_rew_path:
            #read dataframe with precomputed potency
            with open(precomp_rew_path,'rb') as f:
                self.df = pickle.load(f)
            # with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/Full_Expt_w_few_inferences/1_4M_12PDBpeptides_w_esm_PCA_w_rewards.pkl','rb') as f:
            #     self.df = pickle.load(f)
        else:
            pass
            # #read dataframe with 440K peptides and their descriptors 
            # with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/data/443K_alpha_helix_w_ifeat_modlamp_desc.pkl','rb') as f:
            #     self.df = pickle.load(f)

            # self.df['pca_X']=self.X_embedded[:,0][np.where(self.y_all==0)]
            # self.df['pca_Y']=self.X_embedded[:,1][np.where(self.y_all==0)]    
            # self.df.pca_X = pd.to_numeric(self.df.pca_X, downcast='float')
            # self.df.pca_Y = pd.to_numeric(self.df.pca_Y, downcast='float')

            # self.df['pca_X16'] = self.df.pca_X.astype(np.float16)
            # self.df['pca_Y16'] = self.df.pca_Y.astype(np.float16)
        self.pep_coord_reward_dict = dict()
        for i in tqdm(range(len(self.df))):
            key = (self.df.iat[i,self.df.columns.get_loc('pca_X16')],self.df.iat[i,self.df.columns.get_loc('pca_Y16')]) #(pcax16, pcay16)
            value = [self.df.iat[i,self.df.columns.get_loc('peptides')],self.df.iat[i,self.df.columns.get_loc('pred_potency')]] #(seq, rf_pred)
            self.pep_coord_reward_dict[key]=value
        print('len pep_coord_reward_dict ', len(self.pep_coord_reward_dict))


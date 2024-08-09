import pandas as pd
import pickle
import gym
from gym import spaces, Env
import matplotlib.pyplot as plt
import time
import numpy as np
import random
# import cv2  
import pickle
import pandas as pd
import sys
from math import cos, sin
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import time
import math
from tqdm import tqdm, tqdm_notebook
from scipy.stats import pearsonr
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
import random
import pymc3 as pm
wandb.login()
import warnings
import logging
warnings.filterwarnings("ignore")
import sys
logging.disable(sys.maxsize)
logger = logging.getLogger("pymc3")
logger.propagate = False


#training with 1M
with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce/Inference/B_samp_data/df_eval1_925000_split.pkl','rb') as f:
    df = pickle.load(f)

df = df[df.split==f'eval_1_0']
print('len_df ', len(df))

df['pca_X16']= df.pca_x.astype(np.float16)
df['pca_Y16']= df.pca_y.astype(np.float16)

def get_lambda(x,OldRange=2, NewRange= 0.02):

    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    scaled_lambda = (((x +1 ) * NewRange) / OldRange) + 0.99
    return np.float16(scaled_lambda)

def get_theta(x,OldRange=2, NewRange= np.pi/3):
    #transform (-1,1)  -> (0,np.pi)
    #OldRange = (1-(-1))  
    #NewRange = (np.pi - 0) 
    theta = (((x +1 ) * NewRange) / OldRange) - np.pi/6
    return np.float16(theta)

def translate_scale_rotate_vector(v,theta,lambda_):
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.float16(lambda_*(np.dot(rot,v)))

def distance_reward(x,y,distance_to_goal,distance_max):
    #distance from (10, -10)
#     distance_to_goal = ((10 - x)**2 + (-10 - y)**2) **0.5
    distance_reward = 1 - (distance_to_goal / (distance_max+sys.float_info.epsilon))**0.5
#     distance_reward = 1/(distance_to_goal+sys.float_info.epsilon)
    return distance_reward

def calculate_descriptors(pep,pep_desc_dict):
    if pep in pep_desc_dict:
        modlamp = pep_desc_dict[pep][0]
        ifeat = pep_desc_dict[pep][1]
    else:
        modlamp = get_modlamp_descriptors(pep)
        ifeat = get_ifeat_desc(pep)
        zeros = np.zeros((14-ifeat.shape[0],576))
        ifeat = np.concatenate((ifeat,zeros))
        pep_desc_dict[pep]=[modlamp,ifeat]
    return pep_desc_dict

def get_Regression_Reward(model_BC,norm_BC,norm_BC_label,descriptor_modlamp, descriptor_ifeat):   
    descriptor_norm = norm_BC.transform(descriptor_modlamp)
    descriptor_ifeat_padded = descriptor_ifeat
    X_ifeat = torch.unsqueeze(torch.Tensor(descriptor_ifeat_padded),dim=0)
    X_ifeat = torch.unsqueeze(X_ifeat,dim=0).float().to(device)
    X_modlamp= torch.Tensor(descriptor_norm).float().to(device)
    with torch.no_grad():
        regression_Score_BC = model_BC(X_ifeat,X_modlamp)
    regression_Score_BC = norm_BC_label.inverse_transform(regression_Score_BC.cpu().detach().reshape(-1,1))
    regression_reward = regression_Score_BC[0]#/max(0.1,regression_Score_NC[0])
    return regression_reward


class PeptideDataframe():
    def __init__(self,pca_path=None,pdb_helices_path=None,precomputed_rewards=True):
        self.df = df
#         #Read peptide PCA
#         with open('./../../../ACP_AMP_project/data/desc_pca_X_embedded_y_all.pkl','rb') as f:
#             self.X_embedded,self.y_all = pickle.load(f)
            
#         if precomputed_rewards:
#             #read dataframe with 1_4M_12PDBpeptides and their descriptors
#             with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/Full_Expt_w_few_inferences/1_4M_12PDBpeptides_w_esm_PCA_w_rewards.pkl','rb') as f:
#                 self.df = pickle.load(f)
            
#         else:
#             #read dataframe with 440K peptides and their descriptors 
#             with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/ACP_AMP_project/data/443K_alpha_helix_w_ifeat_modlamp_desc.pkl','rb') as f:
#                 self.df = pickle.load(f)

#             self.df['pca_X']=self.X_embedded[:,0][np.where(self.y_all==0)]
#             self.df['pca_Y']=self.X_embedded[:,1][np.where(self.y_all==0)]    
#             self.df.pca_X = pd.to_numeric(self.df.pca_X, downcast='float')
#             self.df.pca_Y = pd.to_numeric(self.df.pca_Y, downcast='float')

#             self.df['pca_X16'] = self.df.pca_X.astype(np.float16)
#             self.df['pca_Y16'] = self.df.pca_Y.astype(np.float16)
        self.pep_coord_reward_dict = dict()
        for i in tqdm_notebook(range(len(self.df))):
            key = (df.iat[i,df.columns.get_loc('pca_X16')],df.iat[i,df.columns.get_loc('pca_Y16')]) #(pcax16, pcay16)
            value = [self.df.iat[i,2],self.df.iat[i,6]] #(seq, rf_pred)
            self.pep_coord_reward_dict[key]=value
        print('len pep_coord_reward_dict ', len(self.pep_coord_reward_dict))

class BuildingEnv(Env):
    def __init__(self,episode_num=0):
        super(BuildingEnv,self).__init__()
        self.observation_space = spaces.Box(low = -35, high = 50, shape=(1,5), 
                                            dtype = dtype)
        self.episode_num = episode_num
        self.pep_desc_dict = dict()
        self.start_model_checkpointing= not False
        # Permissible area of helicper to be 
        self.y_min = -35
        self.x_min = -35
        self.y_max = 45
        self.x_max = 50
        # Define an action space (\lambda \in (0,1], \theta \in [0,1)  -> to be later transformed to (0,2*pi))
        self.action_space = spaces.Box(low = 0, high= +1, shape= (2,), dtype=dtype)
        self.action_space = spaces.Box(low = -1, high =1, shape = (2,), dtype=dtype)
        self.cur_goal = np.array([10,-10],dtype=dtype)
        self.max_bc_score = 0.0
        pep_df = PeptideDataframe()
        self.df = pep_df.df
        self.all_unique_encountered_peptides = set()
        self.pep_coord_reward_dict = pep_df.pep_coord_reward_dict
        self.goal_to_potency_dict=dict()
        self.goal_update_count = 0
        self.cur_hotspot_mu = 0
        self.new_goals = set()
        
        if self.episode_num!=0:
            self.cur_hotspot_mu, self.cur_goal = 0,None
            self.max_bc_score = 0.7379133948565993 #manually defined by looking at failed run's console log
            self.all_unique_encountered_peptides = all_unique_encountered_peptides
            self.goal_to_potency_dict = goal_to_potency_dict
            for k in goal_to_potency_dict.keys():
                if goal_to_potency_dict[k][-1][0] > self.cur_hotspot_mu:
                    self.cur_hotspot_mu = goal_to_potency_dict[k][-1][0]
                    self.cur_goal = np.array(k,dtype =dtype)
        
    def reset(self):
        if (self.start_model_checkpointing) and (self.episode_num%100==0):
            print(f'Saving.....{self.episode_num}')
            model.save(f"{models_dir}/Driver1Mretry_{self.episode_num}")
            print('Peptides encountered so far ',len(self.all_unique_encountered_peptides))
            with open(f'{models_dir}/Driver1Mretry_{self.episode_num}_all_peps.pkl','wb') as f:
                pickle.dump(self.all_unique_encountered_peptides,f)
#         TODO
        if (self.episode_num >0) & (self.episode_num%50==0):
            with open(f'{models_dir}/Driver1Mretry_{self.episode_num}_goal_to_potency_dict.pkl','wb') as f:
                pickle.dump(self.goal_to_potency_dict,f)
            self.cur_goal = self.thompson_choose_goal()
            self.goal_update_count = 0
            self.new_goals = set()
            self.new_goals.add((self.cur_goal[0],self.cur_goal[1])) #add goal chosen by Thompson into new_goals dict, so that it participates in next round of MCMC estimation.
            
        self.episode_num +=1
        self.step_ctr=0
        print(f'Starting new episode# {self.episode_num}') 
        self.x = random.uniform(self.x_min,self.x_max)
        self.y = random.uniform(self.y_min,self.y_max)
        self.distance_to_goal = ((self.cur_goal[0] - self.x)**2 + (self.cur_goal[1] - self.y)**2) **0.5
        self.distance_max = self.distance_to_goal # When episode begins, the max distance is starting point to current goal.
        start_state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=dtype)
        self.state = start_state
        print(f'Agent starting location {self.state}, goal {self.cur_goal}')
        self.episode_encountered_peptides=set()
        self.pos_x=[self.x]
        self.pos_y=[self.y]
        self.pos = set((self.x,self.y))
        self.logging_str = ''
        
        return self.state
    
    def render(self,mode = 'human'):
        pass
    
    def update_goal(self, regression_score):
#         print(f'reg score {regression_score}, max_bc_score {self.max_bc_score}, {regression_score> self.max_bc_score}')
        if (regression_score> 0.9*self.max_bc_score) and ((self.x,self.y) not in self.goal_to_potency_dict) :#or ((self.start_10per_scheme) and (regression_score>self.cur_hotspot_mu)):
            self.max_bc_score = max(regression_score,self.max_bc_score)
            self.cur_goal[0], self.cur_goal[1]= self.x,self.y # 10,-10
            self.state[2]=0.0
            self.state[3],self.state[4]= self.cur_goal[0],self.cur_goal[1]
            print(f'goal update {self.state}')
            self.goal_to_potency_dict[(self.cur_goal[0],self.cur_goal[1])]=[set(),[],[]] # [set(peptides),[regressionScores], [estimated mu, estimated_sigma]]
            self.goal_update_count +=1
            self.new_goals.add((self.cur_goal[0],self.cur_goal[1]))  
     

    def get_posterior(self,regression_scores):
        '''MCMC to estimate \mu of regression score generating normal distribution 
        '''
        with pm.Model() as model:

            prior_mu = pm.Normal('mu', mu=0, sigma=100)  # prior_mu
            prior_sigma = pm.HalfNormal('sigma', sigma =1) #prior_sigma
            
            obs = pm.Normal('obs', mu=prior_mu, sigma=prior_sigma, observed=regression_scores)  # likelihood
            step = pm.Metropolis()

            # sample with 3 independent Markov chains;
            trace = pm.sample(draws=50000, chains=1, step=step, return_inferencedata=True, progressbar=False)  
        return trace
    
    def thompson_choose_goal(self):
        # https://gdmarmerola.github.io/approximate-bayes-bandits/
        #https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/getting_started.html
        # choice of bandit
        best_goal_avg = 0.0
        posterior_mu_list = []
        if self.goal_update_count>0:
            'if goal has updated once or more between 2 Thompson periods (50 episodes), recalculate posteriors for all cluster centers'
            for k in tqdm_notebook(self.goal_to_potency_dict):
                #Running MCMC for 3x50000x#peptides will run out of memory as #peptides for that goal point grows too large.
                #Hence, run for latest 2000 peptides for that goal point
                if k in self.new_goals:
                    last_2k_peps = self.goal_to_potency_dict[k][1][-min(2000,len(self.goal_to_potency_dict[k][1])):]
                    trace = self.get_posterior(last_2k_peps)
                    posterior_mu = trace.posterior['mu'].mean().values
                    posterior_sigma = trace.posterior['sigma'].mean().values
                    self.goal_to_potency_dict[k][2] = [posterior_mu,posterior_sigma]
        else:
            'do mcmc only with cur_goal'
            k = tuple(self.cur_goal)
            #Running MCMC for 3x50000x#peptides will run out of memory as #peptides for that goal point grows too large.
            #Hence, run for latest 2000 peptides for that goal point
            last_2k_peps = self.goal_to_potency_dict[k][1][-min(2000,len(self.goal_to_potency_dict[k][1])):]
            trace = self.get_posterior(self.goal_to_potency_dict[k][1])
            posterior_mu = trace.posterior['mu'].mean().values
            posterior_sigma = trace.posterior['sigma'].mean().values
            self.goal_to_potency_dict[k][2] = [posterior_mu,posterior_sigma]
            
            
        for k in self.goal_to_potency_dict:
            posterior_mu, posterior_sigma = self.goal_to_potency_dict[k][-1]
            posterior_mu_list.append(posterior_mu)
            posterior_sample = np.random.normal(posterior_mu,posterior_sigma)
            print(k, (posterior_mu,posterior_sigma), posterior_sample)
            if posterior_sample>best_goal_avg:
                best_goal_avg = posterior_sample
                max_goal = k
                self.cur_hotspot_mu = posterior_mu
                
        print(' Sampled Goal with Highest Avg Potency', max_goal, ' Sampled Potency ', best_goal_avg )
        return np.array([max_goal[0],max_goal[1]],dtype=dtype) 

#     def thompson_choose_goal(self):
#         # https://gdmarmerola.github.io/approximate-bayes-bandits/
#         # choice of bandit
#         max_goal = (self.cur_goal[0],self.cur_goal[1])
#         best_goal_avg = 0.0
#         for goal in tqdm_notebook(self.goal_to_potency_dict):
#             regression_list = self.goal_to_potency_dict[goal][1]
#             trace = self.get_posterior(regression_list)
#             posterior = trace.posterior['mu'].mean().values
#             if posterior>best_goal_avg:
#                 best_goal_avg=posterior
#                 max_goal = goal
#         print('Goal with Highest Avg Potency', max_goal, ' Avg Potency ', best_goal_avg)
#         return np.array([max_goal[0],max_goal[1]],dtype=dtype)  

    
    def isReexplored(self):
        try:
            x_index = self.pos_x.index(self.x)
        except ValueError as e:
            x_index = -1
            
        if x_index!=-1:
            if math.isclose(self.y,self.pos_y[x_index],abs_tol=1e-1):
                return True
        return False
   
    def step(self,action): #TODO: condition to update cur_goal when necessary
        self.step_ctr+=1
        done = False
        reward = -1
#         print(f'action0 {action[0]}  action1 {action[1]}')
        lambda_, theta = get_lambda(np.float16(action[0])), get_theta(np.float16(action[1]))
        
        temp_state_ = translate_scale_rotate_vector(self.state[0:2],theta, lambda_)#.astype(dtype)
        

        #update state        
        self.x, self.y = temp_state_[0], temp_state_[1]
        self.state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=dtype)
        
        #Reexploration Penalty
        if self.isReexplored():
            reward+=0
            
        else:
            self.pos_x.append(self.x)
            self.pos_y.append(self.y)
            self.pos.add((self.x,self.y))
            self.distance_to_goal = ((self.cur_goal[0] - self.x)**2 + (self.cur_goal[1] - self.y)**2) **0.5
            #Matching Peptide at this location
            if (self.x,self.y) in self.pep_coord_reward_dict:
                matching_pep = self.pep_coord_reward_dict[(self.x,self.y)]

                #reward update
                pep = matching_pep[0]
                if pep in self.episode_encountered_peptides:
                    reward+=-10 #reexploration penalty
                else:
                    self.episode_encountered_peptides.add(pep)
    #                 self.pep_desc_dict = calculate_descriptors(pep,self.pep_desc_dict)
    #                 modlamp,ifeat = self.pep_desc_dict[pep][0],self.pep_desc_dict[pep][1]
    #                 regression_score = get_Regression_Reward(model_BC,norm_BC,norm_BC_label,modlamp,ifeat)[0]
                    regression_score = matching_pep[1]/100 #Using precomputed scores for testing speed. Divided by 100 to make it at same scale as distance reward
                    reward+= regression_score
                    self.logging_str+= f'\n Peptide found : seq : {pep}, Regression {reward+1}'
                    self.update_goal(regression_score)
                    
                    #Put encountered peptide and its regression score into goal_to_potency_dict for Thompson sampling later
                    pep_potency = self.goal_to_potency_dict[(self.cur_goal[0],self.cur_goal[1])]
                    if not pep in pep_potency[0]:
                        pep_potency[0].add(pep)
                        pep_potency[1].append(regression_score)


            reward+= distance_reward(self.x,self.y,self.distance_to_goal,self.distance_max)
        
        #termination conditions
        if (self.x<self.x_min) or (self.x>self.x_max) or (self.y<self.y_min) or (self.y>self.y_max):
            done = True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
            reward = -1000 
#             print('ending due to OOB x,y',self.x,self.y)
            
# #         if (self.x!=0) and (self.y!=0) and abs(self.x)<1e-7 and abs(self.y)<1e-7:
#         if abs(self.x)<1e-7 and abs(self.y)<1e-7:
#             reward= -1000
#             done = True
#             self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
#             print('ending due to small x,y',self.x,self.y)
            
        if self.step_ctr>=1000: 
            done = True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
#             if len(self.episode_encountered_peptides)>2:
#                 self.start_model_checkpointing = True
#             reward = avg_reward
            #TODO: add average of all peptides score found in these 10K steps
            print('Episode ended at ',self.x,self.y,self.distance_to_goal)
        
        if (self.x==self.cur_goal[0] )and (self.y==self.cur_goal[1]):
            done=True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
            reward =1000
#             print(f'successful episode; took {self.step_ctr} steps')
        
#         print(f'x {self.x}, y {self.y}, reward {reward}')


        if done:
            print(self.logging_str.split('\n')[-2:])
        return self.state, reward, done, {}

# TODO: Add thompson
#     check if previous training works in inferece setting
    

dtype = np.float16
env = BuildingEnv()
save_iter = 0
TOTAL_TIMESTEPS = 100000000
models_dir = "/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/Reinforcement_Learning/KerasRL_Testing/models/Test"
model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="/home/mkpandey/MyProjects/Generative_Modeling/ACP/Reinforcement_Learning/KerasRL_Testing/a2c_cartpole_tensorboard/",buffer_size=10000000,
            device='cuda:0')
while save_iter< TOTAL_TIMESTEPS:
    model.learn(total_timesteps=100000000,log_interval=5, tb_log_name='Gen12merDriver1M_retry',
                reset_num_timesteps=False)
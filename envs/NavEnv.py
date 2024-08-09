import numpy as np
from gym import Env, spaces
import sys
sys.path.append('/home/mkpandey/MyProjects/Generative_Modeling/ACP/PepSce/PepSce_github')
from data.PeptideDF import PeptideDataframe
import random
import pickle
import pymc3 as pm
import math 
from utils import get_lambda, get_theta, distance_reward, translate_scale_rotate_vector
from tqdm import tqdm

class BuildingEnv(Env):
    def __init__(self,root, dtype = np.float16,episode_num=0):
        super(BuildingEnv,self).__init__()
        pep_df = PeptideDataframe(root+'/data/TARSA_df_train_ESM_PCA_Rew.pkl')
        self.y_min = 1.1*min(pep_df.df.pca_Y16) 
        self.x_min = 1.1*min(pep_df.df.pca_X16) 
        self.y_max = 1.1*max(pep_df.df.pca_Y16)
        self.x_max = 1.1*max(pep_df.df.pca_X16)
        # print(self.x_min, self.x_max, self.y_min, self.y_max)
        self.observation_space = spaces.Box(low = min(self.x_min,self.y_min), high = max(self.x_max,self.y_max), shape=(1,5), 
                                            dtype = dtype)
        self.episode_num = episode_num
        self.dtype=dtype
        self.action_space = spaces.Box(low = 0, high= +1, shape= (2,), dtype=dtype)
        self.action_space = spaces.Box(low = -1, high =1, shape = (2,), dtype=dtype)
        self.cur_goal = np.array([10,-10],dtype=dtype)
        self.max_bc_score = 0.0
        self.df = pep_df.df
        self.all_unique_encountered_peptides = set()
        self.pep_coord_reward_dict = pep_df.pep_coord_reward_dict
        self.goal_to_potency_dict=dict()
        self.goal_update_count = 0
        self.cur_hotspot_mu = 0
        self.new_goals = set()
        self.models_dir = root+"/models/"
        self.model = None

    def set_model(self, model):
        self.model = model

    def reset(self):
        if (self.episode_num%100==0) and (self.model):
            print(f'Saving.....{self.episode_num}')
            self.model.save(f"{self.models_dir}/SAC_Thompson_2_4ipynb_{self.episode_num}")
            print('Peptides encountered so far ',len(self.all_unique_encountered_peptides))
            with open(f'{self.models_dir}/SAC_Thompson_2_4ipynb_{self.episode_num}_all_peps.pkl','wb') as f:
                pickle.dump(self.all_unique_encountered_peptides,f)
#         TODO
        if (self.episode_num >0) & (self.episode_num%50==0):
            with open(f'{self.models_dir}/SAC_Thompson_2_4ipynb_{self.episode_num}_goal_to_potency_dict.pkl','wb') as f:
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
        start_state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=self.dtype)
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
    
    def update_goal(self, regression_score, mode='k'):
#         print(f'reg score {regression_score}, max_bc_score {self.max_bc_score}, {regression_score> self.max_bc_score}')
        if (regression_score> 0.9*self.max_bc_score) and ((self.x,self.y) not in self.goal_to_potency_dict) :
            self.max_bc_score = max(regression_score,self.max_bc_score)
            self.cur_goal[0], self.cur_goal[1]= self.x,self.y 
            self.state[2]=0.0
            self.state[3],self.state[4]= self.cur_goal[0],self.cur_goal[1]
            print(f'goal update {self.state}')
            self.goal_to_potency_dict[(self.cur_goal[0],self.cur_goal[1])]=[set(),[],[]]
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
            for k in tqdm(self.goal_to_potency_dict):
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
        return np.array([max_goal[0],max_goal[1]],dtype=self.dtype) 
    
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
        self.state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=self.dtype)
        
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

# from stable_baselines3 import DDPG, SAC, PPO
# env = BuildingEnv(root='/home/mkpandey/MyProjects/Generative_Modeling/ACP/PepSce')
# save_iter = 0
# TOTAL_TIMESTEPS = 100000000
# model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="/home/mkpandey/MyProjects/Generative_Modeling/ACP/Reinforcement_Learning/KerasRL_Testing/a2c_cartpole_tensorboard/",buffer_size=10000000,
#             device='cuda:0')
# while save_iter< TOTAL_TIMESTEPS:
#     model.learn(total_timesteps=100000000,log_interval=5, tb_log_name='test',
#                 reset_num_timesteps=False)
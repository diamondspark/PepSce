from NavEnv import BuildingEnv
import numpy as np
import pickle
import random
from utils import get_lambda, get_theta, distance_reward, translate_scale_rotate_vector

class InferenceEnv(BuildingEnv):
    def __init__(self,root, dtype = np.float16,episode_num=0):
        super().__init__(root, dtype,episode_num)
        print(self.x_min, self.x_max, self.y_min, self.y_max)
        # goal_to_mu_path
        with open (goal_to_mu_path,'rb') as f: 
            self.goal_to_potency_dict = pickle.load(f)
            
        self.goal_to_mu_dict = dict()
        for k in self.goal_to_potency_dict.keys():
            self.goal_to_mu_dict[k] = [np.mean(self.goal_to_potency_dict[k][1]),np.std(self.goal_to_potency_dict[k][1])]
        
        self.pep_coord_reward_dict = pep_coord_reward_dict
        self.circle=True
        self.circle_radius = 1
        self.circle_threshold = 0.50
        self.pep_coords = list(self.pep_coord_reward_dict.keys())
        self.accepted_coords = set()
        
        self.cur_1000_div = 0
        self.cur_goal = np.array(self.sample_goal(self.goal_to_mu_dict, self.circle_threshold),dtype=dtype)
        self.circle_centers = set()
        self.all_unique_encountered_peptides_rewardlist=[]
        self.screening_log = f'{root}/logs/screening_log_samplemu.txt'
        with open(self.screening_log,'w') as f:
            f.write('\n')

        def sample_goal(self, goal_to_mu_dict, circle_threshold):
            sample_list, goal_list =[], list(goal_to_mu_dict.keys())
            mu_list = []
            for k in goal_to_mu_dict.keys():
                mu, sigma = goal_to_mu_dict[k]
                mu_list.append(mu)
                sample_list.append(np.random.normal(mu,sigma))

            max_mu_id = np.argmax(sample_list)
            return goal_list[max_mu_id]
        
    def reset(self):
        if (self.episode_num >0) & (self.episode_num%100==0):
            print(f' ep# {self.episode_num} {len(self.all_unique_encountered_peptides)} unique peptides discovered so far with average potency {sum(self.all_unique_encountered_peptides_rewardlist)/len(self.all_unique_encountered_peptides_rewardlist)}')
            with open(self.screening_log,'a') as f:
                f.write(f' ep# {self.episode_num} {len(self.all_unique_encountered_peptides)} unique peptides discovered so far with average potency {sum(self.all_unique_encountered_peptides_rewardlist)/len(self.all_unique_encountered_peptides_rewardlist)}'+'\n')
        
        if (self.episode_num >0) & (self.episode_num%500==0): #10000
            self.cur_goal =  sample_goal(self.goal_to_mu_dict, self.circle_threshold)
            print('New goal sampled', self.cur_goal)
            with open(self.screening_log,'a') as f:
                f.write(f'cur_goal {self.cur_goal} \n')

            self.goal_update_count = 0
            with open(f'{root}/logs/screening_log_samplemu_goal_all_hotspots.pkl','wb') as f:
                pickle.dump([self.cur_goal,self.all_unique_encountered_peptides],f)
            
            
        self.episode_num +=1
        self.step_ctr=0
#         print(f'Starting new episode# {self.episode_num}') 
        self.x = random.uniform(self.x_min,self.x_max)
        self.y = random.uniform(self.y_min,self.y_max)
        self.distance_to_goal = ((self.cur_goal[0] - self.x)**2 + (self.cur_goal[1] - self.y)**2) **0.5
        self.distance_max = self.distance_to_goal # When episode begins, the max distance is starting point to current goal.
        start_state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=dtype)
        self.state = start_state
#         print(f'Agent starting location {self.state}, goal {self.cur_goal}')
        self.episode_encountered_peptides=set()
        self.pos_x=[self.x]
        self.pos_y=[self.y]
        self.pos = set((self.x,self.y))
        self.logging_str = ''
        
        self.agent_location_history.append({'x':[],'y':[]})
        
        self.agent_location_history[-1]['x'].append(self.state[0])
        self.agent_location_history[-1]['y'].append(self.state[1])
        return self.state
        
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

                    regression_score = matching_pep[1]/100 #Using precomputed scores for testing speed. Divided by 100 to make it at same scale as distance reward
                    reward+= regression_score
                    self.logging_str+= f'\n Peptide found : seq : {pep}, Regression {reward+1}'
                    print(f'\n Peptide found : seq : {pep}, Regression {reward+1}')
                    if pep not in self.all_unique_encountered_peptides:
                        self.all_unique_encountered_peptides_rewardlist.append(regression_score)
                    
                    

            reward+= distance_reward(self.x,self.y,self.distance_to_goal,self.distance_max)
        
        #termination conditions
        if (self.x<self.x_min) or (self.x>self.x_max) or (self.y<self.y_min) or (self.y>self.y_max):
            done = True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
            reward = -1000 

            
        if self.step_ctr>=100: 
            done = True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
            if len(self.all_unique_encountered_peptides)//1000>self.cur_1000_div:
                self.cur_1000_div = len(self.all_unique_encountered_peptides)//1000
#                 with open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Cache3/ReinforcementLearning/inferenceResults/Ct_{len(self.all_unique_encountered_peptides)}_goal_highMu_hotspots.pkl','wb') as f:
#                     pickle.dump(self.all_unique_encountered_peptides,f)
#             if len(self.episode_encountered_peptides)>2:
#                 self.start_model_checkpointing = True
#             reward = avg_reward
            #TODO: add average of all peptides score found in these 10K steps
#             print('Episode ended at ',self.x,self.y,self.distance_to_goal)
        
        if (self.x==self.cur_goal[0] )and (self.y==self.cur_goal[1]):
            done=True
            self.all_unique_encountered_peptides=self.all_unique_encountered_peptides|self.episode_encountered_peptides
            reward =1000
#             print(f'successful episode; took {self.step_ctr} steps')
        
#         print(f'x {self.x}, y {self.y}, reward {reward}')


#         if done:
#             print(self.logging_str.split('\n')[-2:])
            
        self.agent_location_history[-1]['x'].append(self.x)
        self.agent_location_history[-1]['y'].append(self.y)
        return self.state, reward, done, {}

            

infenv = InferenceEnv('/home/mkpandey/MyProjects/Generative_Modeling/ACP/PepSce/PepSce_github')

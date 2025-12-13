import numpy as np
from gymnasium import Env, spaces
import pickle
import math
import random
import pymc as pm
import torch
from envs.navenv_utils import get_lambda, get_theta, distance_reward, translate_scale_rotate_vector, calculate_descriptors
from reward.random_forest.rf_inference import get_Regression_Reward_RF, find_best_rf_model, load_rf_model, load_rf_scaler
from tqdm import tqdm
# from gym import Env, spaces

class BuildingEnv(Env):
    def __init__(self, train_nav_data, config, models_save_dir = None, dtype = np.float16, episode_num=0):
        super(BuildingEnv,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        if config.reward_model_type=='rf':
            model_info = find_best_rf_model('./models/random_forest/', verbose=True)
            self.reward_model = load_rf_model(model_info['model_path'])
            self.scaler = load_rf_scaler(model_info['scaler_path'])
        else:
            self.reward_model = torch.load('./models/regression/best_model.pt').to(self.device)
        vals = train_nav_data.nav_2d
        min_val = vals[:, 0].min() 
        max_val = vals[:, 1].max()

        low_bound  = min_val - 0.10 * abs(min_val)
        high_bound = max_val + 0.10 * abs(max_val)

        self.observation_space = spaces.Box(
            low  = low_bound,
            high = high_bound,
            shape=(1,5),
            dtype=dtype
        )

        self.episode_num = episode_num
        self.pep_desc_dict = dict()
        self.start_model_checkpointing= not False
        # Permissible area of agent to be, perhaps get it from peptide dataframe 
        self.y_min = self.x_min = low_bound
        self.y_max = self.x_max = high_bound
        self.models_dir = config.rl.model_savedir
        # self.model = model
        self.dtype = dtype
        # Define an action space (\lambda \in (0,1], \theta \in [0,1)  -> to be later transformed to (0,2*pi))
        # self.action_space = spaces.Box(low = 0, high= +1, shape= (2,), dtype=dtype)
        self.action_space = spaces.Box(low = -1, high =1, shape = (2,), dtype=dtype)
        self.cur_goal = np.array([10,-10],dtype=dtype) #TODO randomly initialize within observation space bounds
        self.max_bc_score = 0.0
        # pep_df = PeptideDataframe()
        # self.df = pep_df.df
        self.all_unique_encountered_peptides = set()
        self.pep_coord_reward_dict = train_nav_data.pep_coord_reward_dict #pep_df.pep_coord_reward_dict
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

  
    def reset(self, seed=None, options=None):
        """
        Gym reset() method:
        - increments episode count
        - saves RL model every 100 episodes
        - optionally saves goal_to_potency_dict every 50 episodes
        - randomizes initial position
        - returns initial observation
        """

        # -----------------------------------------------------
        # Save RL model every 100 episodes
        # -----------------------------------------------------
        # if self.start_model_checkpointing and (self.episode_num % 100 == 0):
        #     if self.model is not None and self.models_dir is not None:
        #         model_path = f"{self.models_dir}/Nav_model_ep_{self.episode_num}.zip"
        #         print(f"[Checkpoint] Saving model → {model_path}")
        #         self.model.save(model_path)

        #     # Save encountered peptides
        #     pep_path = f"{self.models_dir}/enc_peptides_ep_{self.episode_num}.pkl"
        #     with open(pep_path, "wb") as f:
        #         pickle.dump(self.all_unique_encountered_peptides, f)

        #     print(f"Total unique peptides so far: {len(self.all_unique_encountered_peptides)}")

        # -----------------------------------------------------
        # Save potency dict & choose new Thompson goal
        # -----------------------------------------------------
        if (self.episode_num > 0) and (self.episode_num % 100 == 0):
            if self.models_dir is not None:
                dict_path = f"{self.models_dir}/goal_to_potency_dict_ep_{self.episode_num}.pkl"
                print(f'saving goal to potency dict at {dict_path}')
                with open(dict_path, "wb") as f:
                    pickle.dump(self.goal_to_potency_dict, f)

            self.cur_goal = self.thompson_choose_goal()
            self.goal_update_count = 0
            self.new_goals = {(self.cur_goal[0], self.cur_goal[1])} #add goal chosen by Thompson into new_goals dict, so that it participates in next round of MCMC estimation.

        # -----------------------------------------------------
        # Start new episode
        # -----------------------------------------------------
        self.episode_num += 1
        self.step_ctr = 0
        print(f"=== Starting Episode {self.episode_num} ===")

        # Random initial position
        self.x = random.uniform(self.x_min, self.x_max)
        self.y = random.uniform(self.y_min, self.y_max)

        # Compute distance to goal
        self.distance_to_goal = math.sqrt(
            (self.cur_goal[0] - self.x) ** 2 + (self.cur_goal[1] - self.y) ** 2
        )
        self.distance_max = self.distance_to_goal # When episode begins, the max distance is starting point to current goal.

        # Construct observation
        self.state = np.array(
            [self.x, self.y, self.distance_to_goal, self.cur_goal[0], self.cur_goal[1]],
            dtype=self.dtype
        )

        print(f"Agent starts at {self.state}, goal = {self.cur_goal}")

        # Reset episode tracking
        self.episode_encountered_peptides = set()
        self.pos_x = [self.x]
        self.pos_y = [self.y]
        self.pos = {(self.x, self.y)}
        self.logging_str = ""
        super().reset(seed=seed)
        info = {}  # This is the extra information dictionary, you can populate it with useful info if needed
        return self.state, info 
    
    def render(self,mode = 'human'):
        pass
    
    def update_goal(self, regression_score):
        """
        Update the RL navigation goal based on regression score of the peptide found at (x, y).

        A new goal is set when:
        - regression_score exceeds 90% of the maximum observed BC score, AND
        - (x, y) is not already a known goal.
        """

        cur_coord = (self.x, self.y)

        # Condition: strong peptide + new coordinate
        should_update = (
            regression_score > 0.9 * self.max_bc_score
            and cur_coord not in self.goal_to_potency_dict
        )

        if not should_update:
            return  # No update required

        # --- Update max BC score ---
        self.max_bc_score = max(regression_score, self.max_bc_score)

        # --- Update current goal coordinates ---
        self.cur_goal[:] = [self.x, self.y]   # assign to numpy array
        self.state[2] = 0.0                   # distance to goal becomes zero
        self.state[3], self.state[4] = self.cur_goal

        print(f"Goal update → new goal: {self.cur_goal}, state: {self.state}")

        # --- Initialize data structure for this new goal ---
        # Format: { goal_coord : [peptide_set, regression_scores, [post_mean, post_sigma]] }
        self.goal_to_potency_dict[cur_coord] = [set(), [], []]

        # --- Thompson sampling bookkeeping ---
        self.goal_update_count += 1
        self.new_goals.add(cur_coord)

     

    def get_posterior(self, regression_scores, draws=50000, chains=3):
        """
        Estimate posterior distribution of μ and σ for peptide regression scores
        using Bayesian inference and Metropolis–Hastings MCMC.

        Parameters
        ----------
        regression_scores : list or np.ndarray
            Observed regression scores for peptides at the current goal.
        draws : int
            Number of MCMC samples (default: 50,000).
        chains : int
            Number of independent chains (default: 1).

        Returns
        -------
        trace : Posterior samples for μ and σ.
        """
        
        with pm.Model() as model:
            # Priors
            mu = pm.Normal("mu", mu=0, sigma=100)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Likelihood
            pm.Normal("obs", mu=mu, sigma=sigma, observed=regression_scores)

            # Sampler
            step = pm.Metropolis()

            # Run MCMC
            trace = pm.sample(
                draws=draws,
                chains=chains,
                step=step,
                return_inferencedata=True,
                progressbar=False
            )

        return trace


    def thompson_choose_goal(self):
        # https://gdmarmerola.github.io/approximate-bayes-bandits/
        #https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/getting_started.html
        # choice of bandit
        best_goal_avg = 0.0
        posterior_mu_list = []
        if self.goal_update_count>0:
            'if goal has updated once or more between 2 Thompson periods (50 episodes), recalculate posteriors for all cluster centers'
            for k in tqdm(self.goal_to_potency_dict, desc=f'MCMC for {len(self.goal_to_potency_dict)} goals/bandits'):
                #Running MCMC for nchainsx50000x#peptides will run out of memory as #peptides for that goal point grows too large.
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
            #Running MCMC for nchainsx50000x#peptides will run out of memory as #peptides for that goal point grows too large.
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
        truncated = False
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
                regression_score = matching_pep[1]

                if pep in self.episode_encountered_peptides:
                    reward+=-10 #reexploration penalty
                else:
                    if regression_score is None:
                        self.episode_encountered_peptides.add(pep)
                        modlamp, ifeat = calculate_descriptors(pep,max_len=self.config.custom_peptide.max_seq_len)
                        if self.config.reward_model_type=='rf':
                            regression_score = get_Regression_Reward_RF(self.reward_model, self.scaler, modlamp, ifeat)
                        else:
                            regression_score = get_Regression_Reward(self.reward_model,modlamp,ifeat, self.device)[0]
                        self.pep_coord_reward_dict[(self.x,self.y)] = [pep, regression_score] #Cache computed rewards 

                    regression_score = regression_score/100 #TODO: Review scaling factor "100". Needs to be a param or non parametric estimate, not hardcoded
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
            print('ending due to OOB x,y',self.x,self.y)
            truncated = True
            
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
        return self.state, reward, done, truncated, {}


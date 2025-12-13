from envs.NavEnv import BuildingEnv
import numpy as np
import pickle
from envs.navenv_utils import get_lambda, get_theta, translate_scale_rotate_vector, sample_goal, distance_reward, calculate_descriptors
from reward.random_forest.rf_inference import get_Regression_Reward_RF
import math
import os
import random

class SamplingEnv(BuildingEnv):
    """
    Sampling-based environment that uses a pre-computed goal-to-potency dictionary
    for goal selection instead of Thompson sampling.
    """
    
    def __init__(self, goal_to_mu_path, train_nav_data, samp_nav_data, config, screening_filename,
                 models_save_dir=None, dtype=np.float16, 
                 episode_num=0):
        """
        Initialize SamplingEnv
        
        Args:
            goal_to_mu_path: Path to pickled goal-to-potency dictionary
            train_nav_data: Navigation data containing peptide coordinates of the training subset.
            samp_nav_data: Navigation data containing peptide coordinates of the screening subset.
            config: Configuration object
            screening_filename: filename for screening subset fasta file. Used to write the logs
            models_save_dir: Directory to save models (optional)
            dtype: Data type for numpy arrays
            episode_num: Starting episode number
        """
        super().__init__(train_nav_data, config, models_save_dir, dtype, episode_num)
        self.pep_coord_reward_dict = samp_nav_data.pep_coord_reward_dict
        # Load goal-to-potency mapping
        self._load_goal_potency_dict(goal_to_mu_path)
        self.screening_filename = screening_filename

        # Evaluation tracking
        self.log_dir = './logs/screening_out'
        os.makedirs(self.log_dir, exist_ok = True)

        # Initialize goal
        self.cur_goal = np.array(
            self._sample_goal(), 
            dtype=dtype
        )
        
        # Tracking sets
        self.all_unique_encountered_peptides_rewardlist = []
        self.discovered_peps = set()
        
        # Initialize log file
        self._init_log_file()
    
    def _load_goal_potency_dict(self, goal_to_mu_path):
        """Load and process goal-to-potency dictionary."""
        with open(goal_to_mu_path, 'rb') as f:
            goal_to_potency_dict = pickle.load(f)
        
        # Convert to goal-to-mu dictionary with mean and std
        self.goal_to_mu_dict = {}
        for k, v in goal_to_potency_dict.items():
            potencies = v[1]  # Assuming v is [peptides, potencies, ...]
            self.goal_to_mu_dict[k] = [
                np.mean(potencies), 
                np.std(potencies)
            ]
    
    def _sample_goal(self):
        """Sample a goal from the goal-to-mu dictionary."""
        return sample_goal(self.goal_to_mu_dict)
    
    def _init_log_file(self):
        """Initialize log file for tracking progress."""
        log_path = f'{self.log_dir}/screening_log_samplemu_{self.screening_filename}.txt'
        with open(log_path, 'w') as f:
            f.write('Sampling Environment Log\n')
            f.write(f'Initial goal: {self.cur_goal}\n\n')
    
    def _log_progress(self, message):
        """Write progress message to log file."""
        log_path = f'{self.log_dir}/screening_log_samplemu_{self.screening_filename}.txt'
        with open(log_path, 'a') as f:
            f.write(message + '\n')
    
    def _save_checkpoint(self):
        """Save checkpoint with current goal and discovered peptides."""
        checkpoint_path = (
            f'{self.log_dir}/{self.screening_filename}_samplemu_'
            f'{self.episode_num}_goal_all_hotspots.pkl'
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump([
                self.cur_goal, 
                self.all_unique_encountered_peptides
            ], f)
        with open(f'{self.log_dir}/{self.screening_filename}_discoveredPeps.pkl','wb') as f:
            pickle.dump(self.discovered_peps)
    
    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode.
        
        Logs progress every 100 episodes and resamples goal every 500 episodes.
        """
        # Log progress every 100 episodes
        if self.episode_num > 0 and self.episode_num % 10 == 0:
            self._log_episode_stats()
        
        # Resample goal every 500 episodes
        if self.episode_num > 0 and self.episode_num % 500 == 0:
            self._resample_goal()
        
        self.episode_num+=1
        self.step_ctr=0
        self.x = random.uniform(self.x_min,self.x_max)
        self.y = random.uniform(self.y_min,self.y_max)
        self.distance_to_goal = ((self.cur_goal[0] - self.x)**2 + (self.cur_goal[1] - self.y)**2) **0.5
        self.distance_max = self.distance_to_goal # When episode begins, the max distance is starting point to current goal.
        start_state = np.array([self.x,self.y,self.distance_to_goal,self.cur_goal[0],self.cur_goal[1]],dtype=self.dtype)
        self.state = start_state
#         print(f'Agent starting location {self.state}, goal {self.cur_goal}')
        self.episode_encountered_peptides=set()
        self.pos_x=[self.x]
        self.pos_y=[self.y]
        self.pos = set((self.x,self.y))
        self.logging_str =''
        # # Call parent reset
        # state, info = super().reset(seed=seed, options=options)
        
        # Reset episode-specific tracking
        self.episode_encountered_peptides = set()
        
        return self.state, {}
    
    def _log_episode_stats(self):
        """Log statistics about discovered peptides."""
        n_unique = len(self.all_unique_encountered_peptides)
        
        if n_unique > 0 and self.all_unique_encountered_peptides_rewardlist:
            avg_potency = (
                sum(self.all_unique_encountered_peptides_rewardlist) / 
                len(self.all_unique_encountered_peptides_rewardlist)
            )
            
            message = (
                f'ep# {self.episode_num} {n_unique} unique peptides '
                f'discovered so far with average potency {avg_potency:.4f}'
            )
            
            print(message)
            self._log_progress(message)
    
    def _resample_goal(self):
        """Resample goal and save checkpoint."""
        self.cur_goal = self._sample_goal()
        self.goal_update_count = 0
        
        message = f'New goal sampled: {self.cur_goal}'
        print(message)
        self._log_progress(f'cur_goal {self.cur_goal}')
        
        self._save_checkpoint()
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Action to take [lambda, theta]
        
        Returns:
            state: New state
            reward: Reward for this step
            done: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.step_ctr += 1
        done = False
        truncated = False
        reward = -1
        
        # Parse action
        lambda_ = get_lambda(np.float16(action[0]))
        theta = get_theta(np.float16(action[1]))
        
        # Update position
        temp_state = translate_scale_rotate_vector(
            self.state[0:2], theta, lambda_
        )
        self.x, self.y = temp_state[0], temp_state[1]
        
        # Update state
        self.state = np.array([
            self.x, self.y, self.distance_to_goal,
            self.cur_goal[0], self.cur_goal[1]
        ], dtype=self.dtype)
        
        # Check for re-exploration
        if self.isReexplored():
            reward += 0  # No penalty in sampling version
        else:
            # Track new position
            self.pos_x.append(self.x)
            self.pos_y.append(self.y)
            self.pos.add((self.x, self.y))
            
            # Update distance to goal
            self.distance_to_goal = math.sqrt(
                (self.cur_goal[0] - self.x)**2 + 
                (self.cur_goal[1] - self.y)**2
            )
            
            # Check for peptide at this location
            if (self.x, self.y) in self.pep_coord_reward_dict:
                reward += self._process_peptide_encounter()
            
            # Add distance reward
            reward += distance_reward(
                self.x, self.y, 
                self.distance_to_goal, 
                self.distance_max
            )
        
        # Check termination conditions
        done, truncated, reward = self._check_termination(reward)
        
        if done:
            self.all_unique_encountered_peptides |= self.episode_encountered_peptides
            # if self.logging_str:
            #     print(self.logging_str.split('\n')[-2:])
        
        return self.state, reward, done, truncated, {}
    
    def _process_peptide_encounter(self):
        """
        Process peptide found at current location.
        
        Returns:
            reward: Reward for this peptide encounter
        """
        matching_pep = self.pep_coord_reward_dict[(self.x, self.y)]
        pep = matching_pep[0]
        regression_score = matching_pep[1]
        
        # Check if already encountered in this episode
        if pep in self.episode_encountered_peptides:
            return -10  # Re-exploration penalty
        else:
            if regression_score is None:
                self.episode_encountered_peptides.add(pep)
                modlamp, ifeat = calculate_descriptors(pep,max_len=self.config.custom_peptide.max_seq_len)
                if self.config.reward_model_type=='rf':
                    regression_score = get_Regression_Reward_RF(self.reward_model, self.scaler, modlamp, ifeat)
                else:
                    regression_score = get_Regression_Reward(self.reward_model,modlamp,ifeat, self.device)[0]
                self.pep_coord_reward_dict[(self.x,self.y)] = [pep, regression_score] #Cache computed rewards 
                self.discovered_peps.add((pep, regression_score))
        
        # Get regression score (pre-computed, divided by 100 for scaling)
        regression_score = regression_score/ 100

        # Log discovery
        self.logging_str += (
            f'\nPeptide found: seq: {pep}, '
            f'Regression: {regression_score:.4f}'
        )
        
        # Track reward for global statistics
        if pep not in self.all_unique_encountered_peptides:
            self.all_unique_encountered_peptides_rewardlist.append(
                regression_score
            )
        
        return regression_score
    
    def _check_termination(self, reward):
        """
        Check if episode should terminate.
        
        Returns:
            done: Whether episode is finished
            truncated: Whether episode was truncated
            reward: Modified reward
        """
        done = False
        truncated = False
        
        # Out of bounds
        if (self.x < self.x_min or self.x > self.x_max or 
            self.y < self.y_min or self.y > self.y_max):
            done = True
            truncated = True
            reward = -1000
            # print(f'Episode ended: out of bounds at ({self.x:.2f}, {self.y:.2f})')
        
        # Max steps reached
        elif self.step_ctr >= 100:
            done = True
            # print(f'Episode ended: max steps at ({self.x:.2f}, {self.y:.2f}), '
            #       f'distance: {self.distance_to_goal:.2f}')
        
        # Goal reached (exact match - unlikely in continuous space)
        elif self.x == self.cur_goal[0] and self.y == self.cur_goal[1]:
            done = True
            reward = 1000
            # print(f'Goal reached in {self.step_ctr} steps!')
        
        return done, truncated, reward
    
    def render(self, mode='human'):
        """Render environment (not implemented)."""
        pass
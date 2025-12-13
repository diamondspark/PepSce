# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/src')
from dataset.navData import NavData
from envs.sampEnv import SamplingEnv
import torch
from easydict import EasyDict
import yaml
import re
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/params.yml' #TODO make argparse argument
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    samp_data_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/data/emb_esm1l6_samp_1' #TODO make argparse argument
    goal_to_mu_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/models/tarsa/goal_to_potency_dict_ep_1275.pkl' #TODO make argparse argument
    train_nav_data = NavData(samp_data_path, 
                        pca_load_path= config.pca_load_path,
                        precomputed_nav_data='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/data/train_nav_data.pkl')
    samp_nav_data = NavData(samp_data_path, 
                        pca_load_path= config.pca_load_path,
                        precomputed_nav_data='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/data/samp1_nav_data.pkl')
    env = SamplingEnv(goal_to_mu_path, train_nav_data, samp_nav_data, config, 
                      screening_filename = re.search(r"(samp_\d+)", samp_data_path).group(1))
    if config.rl.model=='SAC':
        model = SAC.load(config.screening.policy_load_path, env=env,
                        device = device)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100000000)

    with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce/Inference/B_samp_data/df_eval1_925000_split.pkl', 'rb') as f:
        df_eval1 = pickle.load(f)
    df_eval1 = df_eval1[df_eval1.split==f'eval_1_{idx}']

    df_eval1['pca_X16']= df_eval1.pca_x.astype(np.float16)
    df_eval1['pca_Y16']= df_eval1.pca_y.astype(np.float16)
    pep_coord_reward_dict = dict()
    for i in tqdm_notebook(range(len(df_eval1))):
        key = (df_eval1.iat[i,df_eval1.columns.get_loc('pca_X16')],df_eval1.iat[i,df_eval1.columns.get_loc('pca_Y16')]) #(pcax16, pcay16)
        value = [df_eval1.iat[i,2],df_eval1.iat[i,6]] #(seq, rf_pred)
        pep_coord_reward_dict[key]=value
        
    import time
    t0 = time.time()
    dtype = np.float16
    env  = BuildingEnv(goal_to_mu_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/Reinforcement_Learning/KerasRL_Testing/models/Test/SAC_Thompson_2_4ipynb_90200_goal_to_potency_dict.pkl')
    model = SAC.load("/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/Reinforcement_Learning/KerasRL_Testing/models/Test/SAC_Thompson_2_4ipynb_90200.zip", env=env,
                    device = 'cuda:0')

    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100000000)
    print(mean_reward, time.time()-t0)


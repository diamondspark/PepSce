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

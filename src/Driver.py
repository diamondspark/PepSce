# import pandas as pd
# import pickle
# import gym
# from gym import spaces, Env
# import matplotlib.pyplot as plt
# import time
# import numpy as np
# import random
# # import cv2  
# import pickle
# import pandas as pd
# import sys
# from math import cos, sin
# import math
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from stable_baselines3 import DDPG, SAC
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# import math
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.datasets import load_boston
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# import time
# import math
# from tqdm import tqdm, tqdm_notebook
# from scipy.stats import pearsonr
# import pandas as pd
# import wandb
# from wandb.integration.sb3 import WandbCallback
# import random
# import pymc3 as pm
# wandb.login()
# import warnings
# import logging
# warnings.filterwarnings("ignore")

# import sys
# logging.disable(sys.maxsize)
# logger = logging.getLogger("pymc3")
# logger.propagate = False

############
# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/src')
import torch
import yaml
import numpy as np
from easydict import EasyDict
from envs.NavEnv import BuildingEnv
import os
from stable_baselines3 import SAC
from utils import split_fasta
from dataset.esm import run_esm_extraction
from dataset.navData import NavData
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path
import glob
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count, set_start_method


def worker(file):
    idx = re.search(r"_(\d+)\.fasta$", file).group(1)
    outdir = f'./data/emb_esm1l6_samp_{idx}'
    print(f"[Worker] Starting ESM extraction for {file}", flush=True)

    run_esm_extraction(
        peptides=file,
        output_dir=outdir
    )

    print(f"[Worker] Finished {file}", flush=True)
    return outdir

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/params.yml' #TODO make argparse argument
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    models_dir = config.rl.model_savedir 
    os.makedirs(models_dir,exist_ok=True)
    # split_fasta(input_fasta='./data/peptides.fasta',
    #             train_size=config.train_batch,
    #             sampling_size=config.samp_batch,
    #             train_out=f'./data/train_{config.train_batch}.fasta',
    #             sample_prefix=f'./data/samp_{config.samp_batch}')
    # run_esm_extraction(train_peptides=f'./data/train_{config.rl.train_batch}.fasta',
    #                    output_dir = "./data/emb_esm1l6")
    train_nav_data = NavData('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/data/emb_esm1l6', 
                        pca_load_path= config.pca_load_path,
                         pca_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/models/train_pcamodel.pkl',
                         precomputed_nav_data='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/data/train_nav_data.pkl')

    env = BuildingEnv(train_nav_data, config, dtype = np.float16, episode_num=0)
    save_iter = 0
    TOTAL_TIMESTEPS = config.rl.total_timesteps
    checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./models/tarsa/checkpoints/',
    name_prefix=f'{config.rl.model}_model',
    save_replay_buffer=True,
    verbose=1
)

    model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./",buffer_size=10000000,
                device=device)
    # while save_iter< TOTAL_TIMESTEPS:
    model.learn(total_timesteps=TOTAL_TIMESTEPS,log_interval=5, 
                tb_log_name='Gen12merDriver2M',
                reset_num_timesteps=False,
                callback = checkpoint_callback)

    if config.tarsa_screening:
        set_start_method("spawn", force=True)
        samp_fasta_files = glob.glob('./data/samp_*.fasta')
        print(f"Found {len(samp_fasta_files)} sampling FASTA files")
        num_workers = config.screening.num_workers if config.screening.num_workers else min(len(samp_fasta_files), cpu_count())  # cap parallelism

        with Pool(processes=num_workers) as pool:
            pool.map(worker, samp_fasta_files)

        # SEQUENTIAL 
        # for _,file in tqdm(enumerate(samp_fasta_files)):
        #     idx = re.search(r"_(\d+)\.fasta$", file).group(1)
        #     run_esm_extraction(peptides=file,
        #                        output_dir=f'./data/emb_esm1l6_samp_{idx}')


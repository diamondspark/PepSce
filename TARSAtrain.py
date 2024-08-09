import pandas as pd
from tqdm import tqdm
from Descriptors import get_modlamp_descriptors, get_ifeat_desc
import pickle
from OracleProxy import Oracle, MyACPDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from data.PeptideDF import PeptideDataframe
import argparse
from envs.NavEnv import BuildingEnv
from stable_baselines3 import DDPG, SAC, PPO
tqdm.pandas()


def train(args):
    env = BuildingEnv(root=args.root)
    save_iter = 0
    TOTAL_TIMESTEPS = 100000000
    model = SAC("MlpPolicy", env, verbose=1,tensorboard_log=args.root+"/logs/",buffer_size=10000000,
                device=args.device)
    # Set the model in the environment
    env.set_model(model)
    while save_iter< TOTAL_TIMESTEPS:
        model.learn(total_timesteps=100000000,log_interval=5, tb_log_name='TARSA',
                    reset_num_timesteps=False)
        
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some peptide sequences.")

    # Add arguments
    parser.add_argument("--device", type=str, required=True, help="Specify the device to use (e.g., 'CPU' or 'CUDA').")
    parser.add_argument("--root", type=str, required=True, help="Path to PepSce")

    # Parse the arguments
    args = parser.parse_args()
    # print(args.device)
    # print(args.root)
    train(args)

if __name__ == "__main__":
    main()

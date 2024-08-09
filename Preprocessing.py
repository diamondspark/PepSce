import pickle
import pandas as pd
import torch
from ESM import ESMEmbed
import argparse
from tqdm import tqdm
from PCA import pca
tqdm.pandas()
from Descriptors import get_modlamp_descriptors, get_ifeat_desc
from OracleProxy import Oracle, MyACPDataset
from torch.utils.data import DataLoader
import numpy as np

def read_peptides(path):
    print('reading file....')
    with open(path, 'r') as file:
        peptides = [line.strip() for line in file]
    df_new = pd.DataFrame({'peptides': peptides})
    return df_new#.head(100000)

def compute_ESM_embedding(df, device):
    esm_helper = ESMEmbed(device)
    print(f'Calculating ESM Embeddings for {len(df)} peptides ....')
    df['ESM'] = df.peptides.progress_apply(lambda x:esm_helper.get_esm(x,False,6).detach().cpu().numpy()) #.iat[i,-1] = get_esm(pep,False,6).detach().cpu().numpy()
    return df

def precompute_rewards(df_train, model, device):
    df_train['modlamp'] = df_train.peptides.progress_apply(lambda x: get_modlamp_descriptors(x))
    df_train['ifeat'] = df_train.peptides.progress_apply(lambda x: get_ifeat_desc(x))
    df_train.dropna(inplace=True)
    print(df_train.shape)
    test_dataset = MyACPDataset(df_train.ifeat.values, np.vstack(df_train.modlamp.tolist()),np.zeros(len(df_train)))
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model.eval()
    y_pred = []
    for inputs, _ in tqdm(test_loader):
        [X_ifeat,X_modlamp] = inputs
        X_ifeat = torch.unsqueeze(X_ifeat,dim=1).float().to(device)
        X_modlamp= X_modlamp.squeeze().float().to(device)
        # print('X_ifeat',X_ifeat.shape,'X_modlamp',X_modlamp.shape)
        with torch.no_grad():
            logits = model(X_ifeat,X_modlamp)
        y_pred.extend(logits.cpu().numpy())        
    y_pred = np.array(y_pred)
    df_train['pred_potency'] = y_pred
    # print(df_train)
    return df_train


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some peptide sequences.")

    # Add arguments
    parser.add_argument("--device", type=str, required=True, help="Specify the device to use (e.g., 'CPU' or 'CUDA').")
    parser.add_argument("--path", type=str, required=True, help="Path to the peptides.txt file.")
    parser.add_argument("--debug", type=bool, required=False, help = "Debug mode")
    parser.add_argument("--Btrain", type=int, required=True, help="Btrain : # samples to train TARSA and fit PCA")
    parser.add_argument("--oraclepath", type=str, required=True, help=" path of trained oracle model")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    print(f"Device: {args.device}")
    print(f"Path to peptides.txt: {args.path}")
    print(f"Debug: {args.debug}")
    print(f"Btrain: {args.Btrain}")
    print(f"oraclepath: {args.oraclepath}")

    # Read peptides from the specified path
    df = read_peptides(args.path)
    
    if args.debug:
        with open(args.path.replace('peptides.txt', 'peptides_df_ESM.pkl'),'rb') as f:
            df = pickle.load(f)
    else:
        # Compute ESM embedding
        df = compute_ESM_embedding(df, torch.device(args.device))
        with open(args.path.replace('peptides.txt', 'peptides_df_ESM.pkl'),'wb') as f:
            pickle.dump(df,f)

    #Compute PCA
    df = pca(df,args.Btrain, args.path) 
    print(df)

    #Precompute external reward (predicted potency) on Btrain samples. 
    #Use same samples as used for fitting PCA
    #model: trained oracle proxy
    df_train = df.sample(n=args.Btrain, random_state=42)
    model = Oracle()
    model.load_state_dict(torch.load(args.oraclepath))
    model.to(args.device)
    print(f'PreComputing Potency using Oracle proxy for {len(df_train)} samples....')
    df_train = precompute_rewards(df_train,model,args.device)
    df_train = df.merge(df_train, on='peptides',how='inner', suffixes=('', '_y'))
    df_train.drop(df_train.filter(regex='_y$').columns, axis=1, inplace=True)
    print(df_train)
    with open(args.path.replace('peptides.txt', 'TARSA_df_train_ESM_PCA_Rew.pkl'),'wb') as f:
        pickle.dump(df_train,f)

if __name__ == "__main__":
    main()
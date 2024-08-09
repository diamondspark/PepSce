import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import argparse
import pickle
import os
from Descriptors import get_modlamp_descriptors, get_ifeat_desc
import torch.nn.functional as F

class ACP_REG_CNN(nn.Module):
    def __init__(self, ppsize=100, feasize=576, conv1_w=3, conv1_s=1, conv1_f=32, pool1_w=32, pool1_s=1, conv2_w=3, conv2_s=1,
                 conv2_f=32,
                 pool2_w=16, pool2_s=1, n_hid=128, hid_func='ReLU'):
        super(ACP_REG_CNN, self).__init__()
        self.ppsize = ppsize
        self.feasize = feasize
        self.conv1_w = conv1_w
        self.conv1_s = conv1_s
        self.conv1_f = conv1_f
        self.pool1_w = pool1_w
        self.pool1_s = pool1_s
        self.conv2_w = conv2_w
        self.conv2_s = conv2_s
        self.conv2_f = conv2_f
        self.pool2_w = pool2_w
        self.pool2_s = pool2_s
        self.n_hid = n_hid
        self.hid_func = hid_func
        self.layer1_out = (self.ppsize + (self.conv1_w // 2 * 2) - self.conv1_w) // self.conv1_s + 1
        self.layer2_out = (self.layer1_out + (self.pool1_w // 2 * 2) - self.pool1_w) // self.pool1_s + 1
        self.layer3_out = (self.layer2_out + (self.conv2_w // 2 * 2) - self.conv2_w) // self.conv2_s + 1
        self.layer4_out = (self.layer3_out + (self.pool2_w // 2 * 2) - self.pool2_w) // self.pool2_s + 1
#         print('layer4',self.layer4_out)
        def conv_layer1(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(1, out_channels=filter_num, kernel_size=(windows, feature_dim), stride=stride_size,padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=False))
        def conv_layer2(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(feature_dim, out_channels=filter_num, kernel_size=(windows, 1), stride=stride_size,
                          padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),nn.LeakyReLU(inplace=False))
        def avg_pool_layer(windows, stride_size):
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=(windows, 1), stride=stride_size, padding=(windows // 2, 0)))
        def max_pool_layer(windows):
#             print('max_pool windoes', windows)
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=(windows, 1), stride=None))
        def fc_layer(input_dim, hidn_num, hid_func):
            act_func = nn.LeakyReLU(inplace=False)
            if hid_func == 'Sigmoid':
                act_func = nn.Sigmoid()
            elif hid_func == 'ReLU':
                act_func = nn.ReLU(inplace=False)
            return nn.Sequential(
                nn.Linear(input_dim, hidn_num, bias=True),
                nn.BatchNorm1d(hidn_num),
                act_func)
        def output_layer(input_dim, output_num):
            return nn.Sequential(
                nn.Linear(input_dim, output_num, bias=False))
        
        self.conv1 = conv_layer1(self.conv1_w, self.feasize, self.conv1_s, self.conv1_f)
        self.avg_pool1 = avg_pool_layer(self.pool1_w, self.pool1_s)
        self.conv2 = conv_layer2(self.conv2_w, self.conv1_f, self.conv2_s, self.conv2_f)
        self.avg_pool2 = avg_pool_layer(self.pool2_w, self.pool2_s)
        self.max_pool = max_pool_layer(self.layer4_out)
        self.fc_layer1 = fc_layer(512, self.n_hid, self.hid_func)
        self.fc_layer4 = fc_layer(128, 64, self.hid_func)
#         self.fc_layer5 = fc_layer(64, 32, self.hid_func)
#         self.output_layer = output_layer(32, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
#         print('avg_pool2',x.shape)
#         x = self.max_pool(x)#.squeeze(3)
        x = x.view(-1,32*16)
#         print('avg_pool2 squeezed',x.shape)
        x = self.fc_layer1(x)
        x = self.fc_layer4(x)
        return x
        x = self.fc_layer5(x)
        x = self.output_layer(x)
        return x#.squeeze(1)
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(139,100)
        self.linear2 = nn.Linear(100,64)
        self.linear3 = nn.Linear(32,1)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.linear1(x)
#         print(x.shape)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.bn2(self.linear2(x))
        x = torch.relu(x)
        return x
        
        return self.linear3(x)
    
class Oracle(nn.Module):
    def __init__(self,*args):
        super(Oracle,self).__init__()
        self.cnn = ACP_REG_CNN()
        self.mlp = MLP()
        self.fc_layer1 = nn.Linear(128, 64)
        self.fc_layer2 = nn.Linear(64, 32)
        self.fc_layer3 = nn.Linear(32,1)
        
    def forward(self,X_ifeat,X_modlamp):
        cnn_output = self.cnn(X_ifeat)
        mlp_output = self.mlp(X_modlamp)
        concat = torch.concat((cnn_output,mlp_output),dim=-1)
#         print('concat',concat.shape)
        lin1 = torch.relu(self.fc_layer1(concat))
        lin2 = torch.relu(self.fc_layer2(lin1))
        output = self.fc_layer3(lin2)
        return output
    
class MyACPDataset(Dataset):
    def __init__(self,X,X_modlamp, y):
        self.ifeatures = X
        self.modlampfeatures=X_modlamp
        self.label = y
    def __getitem__(self, index):
        X = self.ifeatures[index]
        zeropad_ifeat = np.zeros((14,576))
        zeropad_ifeat[:X.shape[0], :X.shape[1]] = X
        fea = [zeropad_ifeat,self.modlampfeatures[index]]
        lab = self.label[index]
        return fea, lab
    def __len__(self):
        return len(self.label)

def train(path, batch_size=32, num_epochs = 500):
    df = pd.read_csv(path)
    df['modlamp'] = df.peptides.apply(lambda x: get_modlamp_descriptors(x))
    df['ifeat'] = df.peptides.apply(lambda x: get_ifeat_desc(x))
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Xmodlamp = np.vstack(df.modlamp.tolist())
    # Initialize lists to store evaluation metrics
    r2_scores = []
    q2_scores = []
    pearson_correlation = []
    spearman_correlation = []

    y = np.array(df.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over each fold
    for train_index, val_index in tqdm(kf.split(Xmodlamp)):
        # Create training and validation datasets
        train_dataset = MyACPDataset(df.ifeat.values[train_index], np.vstack(df.modlamp.tolist())[train_index],y[train_index])
        val_dataset = MyACPDataset(df.ifeat.values[val_index], np.vstack(df.modlamp.tolist())[val_index],y[val_index])

        # Initialize DataLoader for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize the model
        model = Oracle().to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Define learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for inputs, target in train_loader:
                [X_ifeat,X_modlamp] = inputs
                X_ifeat = torch.unsqueeze(X_ifeat,dim=1).float().to(device)
                X_modlamp= X_modlamp.squeeze().float().to(device)
                print('X_ifeat',X_ifeat.shape,'X_modlamp',X_modlamp.shape,)
                target =  target.float().to(device)
                target = target.reshape((target.shape[0], 1))
                optimizer.zero_grad() #Reset gradients 
                logits = model(X_ifeat,X_modlamp)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for inputs, target in val_loader:
                [X_ifeat,X_modlamp] = inputs
                X_ifeat = torch.unsqueeze(X_ifeat,dim=1).float().to(device)
                X_modlamp= X_modlamp.squeeze().float().to(device)
    #             print('X_ifeat',X_ifeat.shape,'X_modlamp',X_modlamp.shape)
                target =  target.float().to(device)
                target = target.reshape((target.shape[0], 1))
                with torch.no_grad():
                    logits = model(X_ifeat,X_modlamp)
                y_pred.extend(logits.cpu().numpy())
                y_true.extend(target.cpu().numpy())
            
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

        # Compute evaluation metrics
        r2 = r2_score(y_true, y_pred)
        r2_scores.append(r2)

        mse = mean_squared_error(y_true, y_pred)
        q2 = 1 - mse / np.var(y_true)
        q2_scores.append(q2)

        pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        pearson_correlation.append(pearson_corr)
        print('Pearson ',pearson_corr)

        spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
        spearman_correlation.append(spearman_corr)
        base_dir = os.path.dirname(path)
        # Change the directory to 'models' and the filename to 'pca.pkl'
        new_dir = os.path.join(base_dir, '../models')
        torch.save(model.state_dict(), new_dir+f'/oracle_pearson_{pearson_corr}.pt')


    # Report the mean scores
    print("Mean R2 Score:", np.mean(r2_scores))
    print("Mean Q2 Score:", np.mean(q2_scores))
    print("Mean Pearson Correlation:", np.mean(pearson_correlation))
    print("Mean Spearman Correlation:", np.mean(spearman_correlation))

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some peptide sequences.")

    # Add arguments
    parser.add_argument("--path", type=str, required=True, help="Path to the DMastoparan.txt file.")

    # Parse the arguments
    args = parser.parse_args()
    train(args.path)

if __name__ == "__main__":
    main()
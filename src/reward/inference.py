import torch

def get_Regression_Reward(model_BC,descriptor_modlamp, descriptor_ifeat, device):   
    descriptor_ifeat_padded = descriptor_ifeat
    X_ifeat = torch.unsqueeze(torch.Tensor(descriptor_ifeat_padded),dim=0)
    X_ifeat = torch.unsqueeze(X_ifeat,dim=0).float().to(device)
    X_modlamp= torch.Tensor(descriptor_modlamp).float().to(device)
    print('reward inference.py ',X_ifeat.shape,X_modlamp.shape)
    with torch.no_grad():
        regression_Score_BC = model_BC(X_ifeat,X_modlamp)
    print(regression_Score_BC)
    regression_reward = regression_Score_BC[0]#/max(0.1,regression_Score_NC[0])
    return regression_reward


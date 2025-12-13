from math import cos, sin
import numpy as np
from dataset.descriptors import get_modlamp_descriptors, get_ifeat_desc
import sys

def get_lambda(x,OldRange=2, NewRange= 0.02):
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    scaled_lambda = (((x +1 ) * NewRange) / OldRange) + 0.99
    return np.float16(scaled_lambda)

def get_theta(x,OldRange=2, NewRange= np.pi/3):
    #transform (-1,1)  -> (0,np.pi)
    #OldRange = (1-(-1))  
    #NewRange = (np.pi - 0) 
    theta = (((x +1 ) * NewRange) / OldRange) - np.pi/6
    return np.float16(theta)

def translate_scale_rotate_vector(v,theta,lambda_):
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.float16(lambda_*(np.dot(rot,v)))

def distance_reward(x,y,distance_to_goal,distance_max):
    #distance from (10, -10)
#     distance_to_goal = ((10 - x)**2 + (-10 - y)**2) **0.5
    distance_reward = 1 - (distance_to_goal / (distance_max+sys.float_info.epsilon))**0.5
#     distance_reward = 1/(distance_to_goal+sys.float_info.epsilon)
    return distance_reward

def calculate_descriptors(pep, max_len=15):
    modlamp = get_modlamp_descriptors(pep)
    ifeat = get_ifeat_desc(pep)
    zeros = np.zeros((max_len-ifeat.shape[0],576))
    ifeat = np.concatenate((ifeat,zeros))
    return modlamp, ifeat

def sample_goal(goal_to_mu_dict):
    sample_list, goal_list =[], list(goal_to_mu_dict.keys())
    mu_list = []
    for k in goal_to_mu_dict.keys():
        mu, sigma = goal_to_mu_dict[k]
        mu_list.append(mu)
        sample_list.append(np.random.normal(mu,sigma))

    max_mu_id = np.argmax(sample_list)
    return goal_list[max_mu_id]
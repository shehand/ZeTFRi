
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

theta = 0.01
phi = 0.5

def compute_model_quality(loss_t_plus_one, loss_t, num_data):
    calculated_data_quality_ratio = abs(loss_t - loss_t_plus_one) * num_data
    powered_ratio = math.pow(calculated_data_quality_ratio, phi)

    tmp_performance_score = None
    if loss_t < loss_t_plus_one:
        tmp_performance_score = 1 - math.exp(theta * powered_ratio)
    else:
        tmp_performance_score = 1 - math.exp(-theta * powered_ratio)
        
    return tmp_performance_score

def compute_model_quality_with_loss_diff(loss_diff, num_data):
    calculated_data_quality_ratio = abs(loss_diff) * num_data
    powered_ratio = math.pow(calculated_data_quality_ratio, phi)

    tmp_performance_score = None
    if loss_diff > 0:
        tmp_performance_score = 1 - math.exp(theta * powered_ratio)
    else:
        tmp_performance_score = 1 - math.exp(-theta * powered_ratio)
        
    return tmp_performance_score

# Static Methods

# A method to convert torch machine learning data dictionary to a python array
def convert_model_dict_to_array(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)
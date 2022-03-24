import torch
import numpy as np
import modman
import json


def Federated_average(list_of_params):
    print("Federated averaging with clients# ", len(list_of_params))
    if(len(list_of_params)<1):
        print("Error gradient list is empty")
        return {}
    if (len(list_of_params)==1):
        return list_of_params[0]
    average_gradient={}
    total_sample=0
    for _,x in list_of_params:
        total_sample += x
        print(x,total_sample)
    for indices in range(len(list_of_params)):
        layer_names=[]
        for x in (list_of_params[indices][0]):
            layer_names.append(x)
        # print(list_of_params[indices][0])    
        print("Layer names", layer_names)    
        for j in range(len(layer_names)):
            sample_size=list_of_params[indices][1]
            list_of_params[indices][0][layer_names[j]]=np.multiply(list_of_params[indices][0][layer_names[j]],sample_size/total_sample)
        
    for indices in range(len(list_of_params)):
        layer_names=[]        
        for x in (list_of_params[indices][0]):
            layer_names.append(x)
        if indices==0:
            for i in range(len(layer_names)):
                # print("params->",list_of_params[indices][0][layer_names[i]].tolist())
                average_gradient[layer_names[i]]=np.array(list_of_params[indices][0][layer_names[i]]).tolist()
            continue
        for i in range(len(layer_names)):            
            average_gradient[layer_names[i]]=np.add(average_gradient[layer_names[i]],list_of_params[indices][0][layer_names[i]]).tolist()
    return average_gradient;

@torch.no_grad()
def FederatedAveragingModel(accu_params: list, global_model: dict) -> dict:
    avg_model = {}
    for key in global_model.keys():
        avg_model[key] = torch.stack(
            [accu_params[i][key].float() for i in range(len(accu_params))], 0).mean(0)

    return avg_model
import requests
import json

from typing import List
import torch

def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_


#################################################################
# Network Handlers
#################################################################

# Collect Client Params

def get_client_params(url: str, model_name: str):

    r = requests.get(url=url+'getclientP/'+model_name)
    if r.status_code != 200:
        print("Server Error: Could not fetch all params.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()
    return data;



# Send Trained Model Gradients (StateDict)
def send_global_model_update(url: str, model_name: str, params: dict):
    body = {'data': {
        'Params': json.dumps(params),
    }
    }

    # Send POST request
    r = requests.post(url=url + 'applyUpdate/' + model_name, json=body)

    if r.status_code != 200:
        print("Server Error: Could not Apply Model Updates.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()
    print( data['reply'])

    return data['reply']

# Fetch Model Lock Status  
def get_update_lock(url: str, model_name: str):
    
    r = requests.get(url=url + 'getUlock/'+model_name)
    if r.status_code != 200:
        print("Server Error: Could not fetch Lock Status.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()
    print("data->", data)
    print("Lock data:->", data['LockStatus'])

    return data['LockStatus'] 



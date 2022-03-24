import modman
import works

import numpy as np
import torch
from time import sleep

# Model Name / ALIAS (in Client)
ALIAS = 'experiment_01'

# API endpoint
URL = "http://localhost:3000/api/model/"

URL = "http://localhost:5500/api/model/"

while True:
    if modman.get_update_lock(URL, ALIAS):
        all_params = modman.get_client_params(URL, ALIAS)
        average_params=works.Federated_average(all_params)
        modman.send_global_model_update(URL,ALIAS,average_params)
        print("Update Complete . . . .")

    sleep(0.02)

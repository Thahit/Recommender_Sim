

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent_dir)
print(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simtrain.train import train_single_function_approx
from simtrain.sim_models_new import all_in_one_model
import pytorch_warmup as warmup
import simtrain.utils as utils
from os.path import join
import simtrain.SETTINGS_POLIMI as SETTINGS
import notebooks.paths as paths
from functools import partial
import csv

# Visualization function
def plot_results(model, sample_path, timecheat, state_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # simpler nn
    simulate_single_partial_forced_function_approx = partial(
        utils.simulate_single_forced_function_approx, path=sample_path,
                                num_tries=1, timecheat=timecheat, state_size=state_size)
    simulate_single_partial_function_approx =partial(
        utils.simulate_single_function_approx, num_events =len(sample_path),
                                num_tries=1, timecheat=timecheat, state_size=state_size)
        
    example_out_forced = simulate_single_partial_forced_function_approx(model)
    example_out = simulate_single_partial_function_approx(model)

    #_________________plot____________________________________--
    time_series_1 = sample_path.detach().numpy() # Timestamps for the first time series
    time_series_2 = (torch.clamp(torch.as_tensor(example_out_forced),0,70).detach().numpy())  # Timestamps for the second time series
    time_series_3 = (torch.clamp(torch.as_tensor(example_out), 0, 70).detach().numpy())  # Timestamps for the second time series

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(time_series_1, [1] * len(time_series_1), color='blue', label='Ground Truth', s=10, marker='o')
    ax.scatter(time_series_2, [2] * len(time_series_2), color='red', label='Simmulation Forced', s=10, marker='x')

    ax.scatter(time_series_3, [3] * len(time_series_3), color='green', label='Simmulation Free', s=10, marker='x')


    # Add labels, legend, and grid
    ax.set_xlabel('Time')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Ground Truth', 'Simmulation Forced', "Simmulation Free"])
    ax.set_title('Comparison of Two Time Series')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Adjust subplot parameters to make room for the legend
    plt.subplots_adjust(right=0.75)
    ax.grid(True)
    plt.savefig(os.path.join(output_dir, f"Plot_samples.png"))
    plt.close()


def save_losses_to_csv(results, output_dir):
    csv_file = os.path.join(output_dir, 'losses.csv')
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        
        for epoch, loss in results:
            writer.writerow([epoch + 1, loss])

def main():
    # Parameters
    num_epochs = 1200
    user_lr_max = 0.001

    state_sizes = [1, 4]
    hidden_dims = [8, 16, 32, 64]
    num_layers1 =  [1, 2, 3, 4]
    num_layers2 =  [1, 2, 3, 4]
    data_points =  [0, 4, 7]
    num_tries_list =  [10, 20]
    output_dir_base = 'experiment_results/function_approx_sampling'

    # Loop over different hyperparameters
    for user in data_points:
        output_dir = os.path.join(output_dir_base, f"user{user}")
        checkpoint = torch.load(join(paths.dat, SETTINGS.rootpaths['models'],
                             "testing", "data.h5"))
        list_of_dicts = checkpoint['data']
        chosen_sample = list_of_dicts[0]["timestamps"]
        sample_path =torch.tensor(chosen_sample)
        for num_tries in num_tries_list:
            for state_size in state_sizes:
                for hidden_dim in hidden_dims:
                    for layers_in_1 in num_layers1:
                        for layers_in_2 in num_layers2:
                            print(f"User: {user}, Testing State Size: {state_size}, hidden dimension: {hidden_dim}, model 1 layers: {layers_in_1}, model 2 layers: {layers_in_2}")
                            
                            # Create and train the model
                            width=hidden_dim
                            user_state_dict = {"model_hyp": {"layer_width": [width for _ in range(3)]}}
                            time_dict = {"model_hyp": {"layer_width": [width for _ in range(3)]}
                                        }

                            timecheat = False
                            hyperparameter_dict = {"state_size": state_size, "time_model": time_dict, 
                                                    "state_model": user_state_dict}
                            model = all_in_one_model(hyperparameter_dict, timecheat=timecheat)

                            steps_per_epoch = 1

                            warmup_period = steps_per_epoch * 10
                            num_steps = num_epochs*steps_per_epoch - warmup_period
                            num_iter_til_first_restart = num_steps + 1

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            print(f"Using device: {device}")

                            optimizer = optim.AdamW(model.parameters(), lr=user_lr_max,
                                                    weight_decay=1e-7)
                            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer, T_0=num_iter_til_first_restart, T_mult=1, eta_min=1e-7)

                            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
                            
                            result_history = train_single_function_approx(model, sample_path, scoring_func=utils.energy_score_loss,
                                state_size=state_size, warmup_scheduler=warmup_scheduler, lr_scheduler=lr_scheduler,
                                optimizer=optimizer, num_epochs=num_epochs, num_tries=num_tries, timecheat=timecheat, loss_print_interval=num_epochs//20,
                                warmup_period=warmup_period)

                            # Save results
                            params = f"size_{state_size}_hidden_{hidden_dim}_u_{user}_m1depth_{layers_in_1}_m2depth_{layers_in_2}_timecheat_{timecheat}_num_tries_{num_tries}"
                            output_dir_exp = os.path.join(output_dir, params)            
                            
                            # creates path?
                            plot_results(model, sample_path, timecheat, state_size, output_dir_exp)

                            save_losses_to_csv(result_history, output_dir_exp)
                            torch.save(model.state_dict(), os.path.join(output_dir_exp, 'model_weights.pth'))
                            
if __name__ == "__main__":
    main()

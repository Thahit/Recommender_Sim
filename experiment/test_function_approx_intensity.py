

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent_dir)
print(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simtrain.train import train_density
from simtrain.sim_models_new import Toy_intensity_Comparer
import pytorch_warmup as warmup
import simtrain.utils as utils
from os.path import join
import simtrain.SETTINGS_POLIMI as SETTINGS
import notebooks.paths as paths
from functools import partial
import csv
from torch.utils.data import Dataset, DataLoader
from simtrain.Dataset import TimestepFrequencyDataset

# Visualization function
def plot_results(model, dataset, state_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # simpler nn
    x_range = np.linspace(0, 71, 200)  # Adjust the range as needed
    x_range_tensor = torch.tensor(x_range, dtype=torch.float32).unsqueeze(1)
    state = torch.zeros((len(x_range), state_size))
    model.eval()
    with torch.no_grad():
        predictions = model(state, x_range_tensor).numpy()
    print(f"area: {np.sum(predictions)*(72/200)}")
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, predictions, label='Model Predictions')

    for i in range(len(dataset)):
        sample = dataset[i]
        x_pos= sample['timestep'].item()
        height = torch.where(sample['frequency']>0, sample['frequency'], .1).item()
        plt.plot([x_pos, x_pos], [0, height], linestyle='--', color='red')

    plt.plot([0, 0], [0, 0], color='red', linestyle='--', alpha=1.0, label='Data Points')

    #plt.scatter(x_train.numpy(), y_train.numpy(), color='red', label='Training Data')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Model Predictions Over Input Range')
    plt.legend()
    plt.grid(True)
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
    num_epochs = 250
    user_lr_max = 0.001

    state_sizes = [1, 2, 4]
    hidden_dims = [8, 16, 32]
    num_layers1 =  [1, 2, 3]
    num_layers2 =  [1, 2, 3]
    data_points =  [0, 4, 7]
    time_embeddings =  [0, 8, 16]
    train_sorted = False
    batchsize = 32
    # Create synthetic data

    # Experiment tracking
    output_dir_base = 'experiment_results/function_approx_intensity'

    # Loop over different hyperparameters
    for user in data_points:
        output_dir = os.path.join(output_dir_base, f"user{user}")  
        checkpoint = torch.load(join(paths.dat, SETTINGS.rootpaths['models'],
                             "testing", "data.h5"))
        list_of_dicts = checkpoint['data']
        chosen_sample = list_of_dicts[0]["timestamps"]
        sample_path =torch.tensor(chosen_sample)
        dataset = TimestepFrequencyDataset(sample_path, num_random_points=150)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=not train_sorted
                        )

        for time_embedding in time_embeddings:
            for state_size in state_sizes:
                for hidden_dim in hidden_dims:
                    for layers_in_1 in num_layers1:
                        for layers_in_2 in num_layers2:
                            print(f"User: {user}, Testing State Size: {state_size}, hidden dimension: {hidden_dim}, model 1 layers: {layers_in_1}, model 2 layers: {layers_in_2}")
                            
                            # Create and train the model
                            width=hidden_dim

                            intensity = {"model_hyp": {"layer_width": [width for _ in range(layers_in_1)]}}
                            state_dict = {"model_hyp": {"layer_width": [width for _ in range(layers_in_2)]},
                                        }

                            hyperparameter_dict = {"state_size": state_size, "state_model": state_dict, 
                                                    "intensity_model": intensity,
                                                    "time_embedding_size" :time_embedding, "max_freq": 40,
                                                    }
                            model = Toy_intensity_Comparer(hyperparameter_dict)

                            steps_per_epoch = len(dataset) // batchsize  
                            if len(dataset) % batchsize  != 0:
                                steps_per_epoch+=1

                            warmup_period = steps_per_epoch * 10
                            num_steps = num_epochs*steps_per_epoch - warmup_period
                            num_iter_til_first_restart = num_steps + 1

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            print(f"Using device: {device}")

                            optimizer = optim.AdamW(model.parameters(), lr=user_lr_max,
                                                    weight_decay=1e-6)
                            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer, T_0=num_iter_til_first_restart, T_mult=1, eta_min=1e-6)

                            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
                            
                            from simtrain.utils import weighted_mse_loss
                            loss = partial(weighted_mse_loss, weight_pos=5)

                            result_history = train_density(model, dataloader, criterion=loss, state_size=state_size,
                                optimizer=optimizer, num_epochs=num_epochs, warmup_scheduler=warmup_scheduler,
                                loss_print_interval=40, warmup_period=warmup_period, lr_scheduler=lr_scheduler)

                            # Save results
                            params = f"size_{state_size}_hidden_{hidden_dim}_u_{user}_m1depth_{layers_in_1}_m2depth_{layers_in_2}_time_embedding_{time_embedding}"
                            output_dir_exp = os.path.join(output_dir, params)            
                            
                            # creates path?
                            plot_results(model, dataset, state_size, output_dir_exp)#change

                            save_losses_to_csv(result_history, output_dir_exp)
                            torch.save(model.state_dict(), os.path.join(output_dir_exp, 'model_weights.pth'))
                            
if __name__ == "__main__":
    main()

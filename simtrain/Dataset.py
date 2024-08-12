from torch.utils.data import Dataset
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of dicts): Each dict contains 'timestamps', 'items', and 'labels' for a user.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_data = self.data[idx]
        timestamps = user_data['timestamps']
        items =  user_data['item_ids']
        labels = user_data['interaction_types']
        return timestamps, items, labels, user_data["user_means"], user_data["user_vars_log"], idx
    
    def Update_user_params(self, means_list, logvar_list, idx_list):
        #means_list.requires_grad = True
        #logvar_list.requires_grad = True
        
        #print(means_list)
        
        self.data[idx_list[0]]["user_means"] = means_list.tolist()
        self.data[idx_list[0]]["user_vars_log"] = logvar_list.tolist()
        #for means, logvar, idx in zip(means_list, logvar_list, idx_list):
        #    self.data[idx]["user_means"] = means
        #    self.data[idx]["user_vars_log"] = logvar



class TimestepFrequencyDataset(Dataset):
    def __init__(self, timesteps, num_random_points=100, interval=0.5, max_time=70,
                 sort_nrs= False):
        """
        Args:
            timesteps (numpy array): Array of positive timesteps.
            num_random_points (int): Number of random points to generate.
            interval (float): Interval length to compute frequency.
            max_time (float): maximum time the dataset is allowed to contain.
        """
        self.timesteps = np.array(timesteps)
        self.num_random_points = num_random_points
        self.interval = interval
        
        # Generate random time points and ensure they include actual timesteps
        self.max_time = max_time
        #self.random_points = np.sort(np.random.uniform(0, self.max_time, self.num_random_points))
        self.random_points = np.linspace(0, max_time, num_random_points)
        self.unique_points = np.unique(np.concatenate([self.random_points, self.timesteps]))
        
        if sort_nrs:
            self.unique_points = np.sort(self.unique_points)
        # Calculate frequencies for each point
        self.frequencies = np.array([self.calculate_frequency(point) for point in self.unique_points])
        
    def calculate_frequency(self, point):
        """
        Calculate frequency of events in the interval [point, point + interval).
        
        Args:
            point (float): The point to calculate frequency for.
        
        Returns:
            float: Frequency of events per interval.
        """
        # could be made more efficient by using the input is sorted
        count = np.sum(np.abs(self.timesteps - point) <= self.interval/2)
        frequency = count / self.interval
        return frequency
    
    def __len__(self):
        return len(self.unique_points)
    
    def __getitem__(self, idx):
        """
        Get item at index `idx`.
        
        Args:
            idx (int): Index of the item to fetch.
        
        Returns:
            dict: A dictionary with 'timestep' and 'frequency'.
        """
        timestep = self.unique_points[idx]
        frequency = self.frequencies[idx]
        return {'timestep': torch.tensor(timestep, dtype=torch.float32),
                'frequency': torch.tensor(frequency, dtype=torch.float32)}

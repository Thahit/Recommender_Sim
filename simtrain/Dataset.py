from torch.utils.data import Dataset


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

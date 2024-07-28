import torch
import torch.nn as nn
from tqdm import tqdm

GRADIENT_CLIPPING_MAX_NORM = 2.

def train_1_path_positive(model, user_state, timestamps, items, labels, loss_func, num_classes, 
                 intensity_loss_func, max_time, device, epsilon = 1e-25,teacher_forcing=True):
    ''' expects batchsize of 1
    '''
    model.init_state(user_state)
    loss_base = 0.
    loss_intensity = 0.
    curr_time = 1e-10# because time 0 is used sadly
    N = len(timestamps)
    max_div_by_N = max_time/N
    for interaction_id in range(N):#[0]
        h = (timestamps[interaction_id] - curr_time).float()
        
        intensity = model.eval_intensity(h, timestamps[interaction_id])+epsilon#ppsilon for stability
        
        #try:
        model.evolve_state(h)
        #except Exception as e:
        #    print("delta: ",h , "\tnew: ", timestamps[interaction_id], "\told: ", curr_time)
        #    print(e)

        curr_time = h
        # no intensity for now
        y_pred = model.view_recommendations(items[interaction_id])# :,
        y_true = torch.as_tensor(labels[interaction_id])# :,
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=num_classes).float()
        
        if teacher_forcing:
            model.jump(y_true_onehot)
        else:
            model.jump(y_pred)
        #loss += loss_func(y_true, y_pred)# for mse
        y_true = y_true.squeeze(0)
        y_pred = y_pred.squeeze(0)
        y_pred = nn.functional.log_softmax(y_pred)
        #print("true: ",y_true.shape, "\t predicted: ",y_pred.shape)
        #print(torch.unique(y_true))
        loss_base += loss_func(y_pred, y_true) # NLLL
        extra_dic = {"max_div_by_N": max_div_by_N}
        loss_intensity += intensity_loss_func(intensity, extra_dic)
    
    return loss_base, loss_intensity

def train_1_path_positive_and_negative(model, user_state, timestamps, items, labels, loss_func, num_classes, 
                 intensity_loss_func, max_time, device, epsilon = 1e-25,teacher_forcing=True,
                 positive_examples_weight=1):
    ''' expects batchsize of 1
    '''
    model.init_state(user_state)
    loss_base = 0.
    loss_intensity = 0.
    curr_time = 1e-10# because time 0 is used sadly
    
    N = len(timestamps)
    #max_div_by_N = max_time/N
    for interaction_id in range(N):#[0]
        h = (timestamps[interaction_id][0] - curr_time).float()# update time
        
        intensity = model.eval_intensity(h, timestamps[interaction_id][0])+epsilon#epsilon for stability
        
        #try:
        model.evolve_state(h)
        #except Exception as e:
        #    print("delta: ",h , "\tnew: ", timestamps[interaction_id], "\told: ", curr_time)
        #    print(e)

        curr_time = h
        if timestamps[interaction_id][1]:# positive example
            positive_id = timestamps[interaction_id][2]
            # no intensity for now
            y_pred = model.view_recommendations(items[positive_id])# :,
            y_true = torch.as_tensor(labels[positive_id])# :,
            y_true_onehot = nn.functional.one_hot(y_true, num_classes=num_classes).float()
            
            if teacher_forcing:
                model.jump(y_true_onehot)
            else:
                model.jump(y_pred)
            #loss += loss_func(y_true, y_pred)# for mse
            y_true = y_true.squeeze(0)
            y_pred = y_pred.squeeze(0)
            y_pred = nn.functional.log_softmax(y_pred)
            #print("true: ",y_true.shape, "\t predicted: ",y_pred.shape)
            #print(torch.unique(y_true))
            loss_base += loss_func(y_pred, y_true) # NLLL
            extra_dic = {}
            loss_intensity += positive_examples_weight * intensity_loss_func(intensity, extra_dic)

        else:# negative example
            loss_intensity += intensity_loss_func(1-intensity)# punish
    return loss_base, loss_intensity


def train(model, device, dataloader,num_epochs, state_size, loss_func, loss_func_kl, optimizer, num_classes, 
            intensity_loss_func, logger, max_time,lr_scheduler, warmup_scheduler,
            kl_weight = 1, user_lr = None, log_step_size = 1, warmup_period=100):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):  # Example: Number of epochs
        loss_all, loss_base, loss_kl, loss_intensity = 0, 0, 0, 0
        #print_user_params(dataloader)# see if values change
        for batch in dataloader:
            # Zero the gradients
            
            timestamps, item_recom, labels, means, logvar, idx = batch
            timestamps, means, logvar = torch.as_tensor(timestamps).to(device).float(), \
                 torch.as_tensor(means).to(device).float(), torch.as_tensor(logvar).to(device).float()  #  item_recom, labels, = item_recom.to(device), labels.to(device),
            
            #timestamps, item_recom, labels, means, logvar = torch.as_tensor(timestamps).to(device), \
            #    torch.as_tensor(item_recom).to(device), torch.as_tensor(labels).to(device), \
            #    torch.as_tensor(means).to(device), torch.as_tensor(logvar).to(device)
            means.requires_grad = True
            logvar.requires_grad = True
            #means.retain_grad()
            #logvar.retain_grad()

            variances = torch.exp(logvar)
            user_state = means + variances*torch.randn((1, state_size))
            #delta_from_previous = torch.cat([torch.zeros((timestamps.size(0),1)), timestamps[:,1:] - timestamps[:,:-1]], dim=1)
            
            curr_loss_base, curr_loss_intensity = train_1_path_positive(model=model, user_state=user_state, 
                            timestamps=timestamps, items=item_recom, labels=labels,  
                         loss_func=loss_func, max_time=max_time, num_classes=num_classes, 
                         device=device, intensity_loss_func=intensity_loss_func)

            curr_loss_kl = kl_weight * torch.sum(loss_func_kl(means, variances))#.view(1,-1)
            
            curr_loss_all = curr_loss_kl + curr_loss_base+curr_loss_intensity
            curr_loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
            optimizer.step()

            #logging
            loss_all += curr_loss_all.item()
            loss_base += curr_loss_base.item()
            loss_kl += curr_loss_kl.item()
            loss_intensity += curr_loss_intensity.item()
            # maybe need to optim user_mean, user_var separate because of torch..
            if user_lr:
                torch.nn.utils.clip_grad_norm_(means, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                torch.nn.utils.clip_grad_norm_(logvar, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                with torch.no_grad():
                    means -= user_lr * means.grad
                    logvar -= user_lr * logvar.grad  
                #print(means)
                means.grad.zero_()
                logvar.grad.zero_()
                dataloader.dataset.Update_user_params(means.detach(), logvar.detach(), idx)
            
            #print("lr: ",optimizer.param_groups[0]['lr'])
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            
            optimizer.zero_grad()
                
        if epoch % log_step_size == 0:
            logger(loss_all, loss_base, loss_kl, loss_intensity)
    logger(loss_all, loss_base, loss_kl, loss_intensity)# log at the end


def train_with_negatives(model, device, dataloader,num_epochs, state_size, loss_func, 
            loss_func_kl, optimizer, num_classes, 
            intensity_loss_func, logger, max_time, lr_scheduler, warmup_scheduler,
            kl_weight=1, user_lr=None, log_step_size=1, warmup_period=100,
            num_negatives=10, positive_examples_weight=1):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):  # Example: Number of epochs
        loss_all, loss_base, loss_kl, loss_intensity = 0, 0, 0, 0
        #print_user_params(dataloader)# see if values change
        for batch in dataloader:
            # Zero the gradients
            
            timestamps, item_recom, labels, means, logvar, idx = batch
            timestamps, means, logvar = torch.as_tensor(timestamps).to(device).float(), \
                 torch.as_tensor(means).to(device).float(), torch.as_tensor(logvar).to(device).float()  #  item_recom, labels, = item_recom.to(device), labels.to(device),
            

            negative_times = torch.FloatTensor(num_negatives).uniform_(0, max_time).to(device)
            all_times = [(t, False, None) for t in negative_times if t not in timestamps]# time, is positive, id of extra data
            all_times.extend([(timestamps[i], True, i) for i in range(len(timestamps))])
            all_times = sorted(all_times, key=lambda x: x[0])

            #timestamps, item_recom, labels, means, logvar = torch.as_tensor(timestamps).to(device), \
            #    torch.as_tensor(item_recom).to(device), torch.as_tensor(labels).to(device), \
            #    torch.as_tensor(means).to(device), torch.as_tensor(logvar).to(device)
            means.requires_grad = True
            logvar.requires_grad = True
            #means.retain_grad()
            #logvar.retain_grad()

            variances = torch.exp(logvar)
            user_state = means + variances*torch.randn((1, state_size))
            #delta_from_previous = torch.cat([torch.zeros((timestamps.size(0),1)), timestamps[:,1:] - timestamps[:,:-1]], dim=1)
            
            curr_loss_base, curr_loss_intensity = train_1_path_positive_and_negative(model=model, user_state=user_state, 
                            timestamps=all_times, items=item_recom, labels=labels,  
                         loss_func=loss_func, max_time=max_time, num_classes=num_classes, 
                         device=device, intensity_loss_func=intensity_loss_func, 
                         positive_examples_weight=positive_examples_weight)

            curr_loss_kl = kl_weight * torch.sum(loss_func_kl(means, variances))#.view(1,-1)
            
            curr_loss_all = curr_loss_kl + curr_loss_base+curr_loss_intensity
            curr_loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
            optimizer.step()

            #logging
            loss_all += curr_loss_all.item()
            loss_base += curr_loss_base.item()
            loss_kl += curr_loss_kl.item()
            loss_intensity += curr_loss_intensity.item()
            # maybe need to optim user_mean, user_var separate because of torch..
            if user_lr:
                torch.nn.utils.clip_grad_norm_(means, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                torch.nn.utils.clip_grad_norm_(logvar, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                with torch.no_grad():
                    means -= user_lr * means.grad
                    logvar -= user_lr * logvar.grad  
                #print(means)
                means.grad.zero_()
                logvar.grad.zero_()
                dataloader.dataset.Update_user_params(means.detach(), logvar.detach(), idx)
            
            #print("lr: ",optimizer.param_groups[0]['lr'])
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            
            optimizer.zero_grad()
                
        if epoch % log_step_size == 0:
            logger(loss_all, loss_base, loss_kl, loss_intensity)
    logger(loss_all, loss_base, loss_kl, loss_intensity)# log at the end




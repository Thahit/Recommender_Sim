import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torchbnn as bnn

GRADIENT_CLIPPING_MAX_NORM = 2.

def train_1_path_positive(model, user_state, timestamps, items, labels, loss_func, 
        num_classes, 
                 intensity_loss_func, max_time, device, epsilon = 1e-25,teacher_forcing=True,
                 conditioned=False):
    ''' expects batchsize of 1
    '''
    if conditioned:
        user_params = torch.clone(user_state)
        model.init_state(user_state, user_params)
    else:
        model.init_state(user_state)
    loss_base = 0.
    loss_intensity = 0.
    curr_time = 1e-10# because time 0 is used sadly
    N = len(timestamps)
    max_div_by_N = max_time/N
    for interaction_id in range(N):#[0]
        h = (timestamps[interaction_id] - curr_time).float()
        
        intensity = model.eval_intensity(h, torch.FloatTensor([curr_time]))#+epsilon#ppsilon for stability
        
        #try:
        model.evolve_state(h)
        #except Exception as e:
        #    print("delta: ",h , "\tnew: ", timestamps[interaction_id], "\told: ", curr_time)
        #    print(e)

        # no intensity for now
        y_pred = model.view_recommendations(items[interaction_id])# :,
        y_pred = nn.functional.log_softmax(y_pred, dim=1)
        y_true = torch.as_tensor(labels[interaction_id])# :,
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=num_classes).float()
        
        if teacher_forcing:
            model.jump(y_true_onehot)
        else:
            model.jump(y_pred)
        #loss += loss_func(y_true, y_pred)# for mse
        y_true = y_true.squeeze(0)
        y_pred = y_pred.squeeze(0)
        #print("true: ",y_true.shape, "\t predicted: ",y_pred.shape)
        #print(torch.unique(y_true))
        loss_base += loss_func(y_pred, y_true) # NLLL
        #print("true: ",y_true, "\t predicted: ",y_pred)
        #print("loss: ", loss_base)
        extra_dic = {"max_div_by_N": max_div_by_N}

        loss_intensity += intensity_loss_func(intensity, extra_dic)
        curr_time = h
    
    return loss_base, loss_intensity


def train_1_path_positive_and_negative(model, user_state, timestamps, items, labels, 
            loss_func, num_classes, 
                 intensity_loss_func, max_time, device, epsilon = 1e-25,teacher_forcing=True,
                 positive_examples_weight=1, conditioned=False):
    ''' expects batchsize of 1
    '''
    if conditioned:
        user_params = torch.clone(user_state)
        model.init_state(user_state, user_params)
    else:
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
        extra_dic = {}
        if timestamps[interaction_id][1]:# positive example
            positive_id = timestamps[interaction_id][2]
            # no intensity for now
            y_pred = model.view_recommendations(items[positive_id])# :,
            y_pred = nn.functional.log_softmax(y_pred, dim=1)
            y_true = torch.as_tensor(labels[positive_id])# :,
            y_true_onehot = nn.functional.one_hot(y_true, num_classes=num_classes).float()
            
            if teacher_forcing:
                model.jump(y_true_onehot)
            else:
                model.jump(y_pred)
            #loss += loss_func(y_true, y_pred)# for mse
            y_true = y_true.squeeze(0)
            y_pred = y_pred.squeeze(0)
            #print(torch.unique(y_true))
            #print(y_pred)
            #print(y_true)
            loss_base += loss_func(y_pred, y_true) # NLLL

            loss_intensity += positive_examples_weight * intensity_loss_func(intensity, extra_dic)

        else:# negative example
            loss_intensity += intensity_loss_func(1-intensity, extra_dic)# punish
    return loss_base, loss_intensity


def train(model, device, dataloader,num_epochs, state_size, loss_func, loss_func_kl, 
          optimizer, num_classes, 
            intensity_loss_func, logger, max_time,lr_scheduler, warmup_scheduler,
            kl_weight = 1, user_lr = None, log_step_size = 1, warmup_period=100, conditioned=False):
    model.to(device)
    #torch.autograd.set_detect_anomaly(True)

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
                         device=device, intensity_loss_func=intensity_loss_func,
                         conditioned=conditioned)

            curr_loss_kl = kl_weight * torch.sum(loss_func_kl(means, variances))#.view(1,-1)
            
            curr_loss_all = curr_loss_kl + curr_loss_base + curr_loss_intensity
            curr_loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
            optimizer.step()

            #for name, param in model.named_parameters():
            #    print(f"Parameter Name: {name}")
            #    print(f"Parameter Value: {param}")
            #    print(f"Gradients: {param.grad}")
            #    print(f"Parameter Shape: {param.shape}")
            #    print(f"Requires Gradient: {param.requires_grad}")
            #    print("-" * 40)
            #return
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


def train_with_negatives(model, device, dataloader,num_epochs, state_size, 
        loss_func, 
            loss_func_kl, optimizer, num_classes, 
            intensity_loss_func, logger, max_time, lr_scheduler, warmup_scheduler,
            kl_weight=1, user_lr=None, log_step_size=1, warmup_period=100,
            num_negatives=10, positive_examples_weight=1, conditioned=False):
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
            #all_times = []
            all_times = [(t, False, None) for t in negative_times if t not in timestamps]# time, is positive, id of extra data
    
            #for t in negative_times:
                # Check if `t` is within the tolerance of any timestamp
            #    if torch.any(torch.abs(timestamps - t) <= 1e-4):
            #        all_times.append((t, False, None))

            #print("num_negatives: ", len(all_times))
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
                         positive_examples_weight=positive_examples_weight, conditioned=conditioned)

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


def train_single_function_approx(model, path, scoring_func,optimizer, state_size, 
                                 lr_scheduler=None, warmup_period=None, warmup_scheduler=None,
                                 num_epochs=100, 
                                 num_tries=20, timecheat=False, loss_print_interval=1):
    results =  []
    for iter in tqdm(range(num_epochs)):
        last_t = 0
        state = torch.zeros((1, state_size))
        loss = 0.

        for timestep in path:
            current_pred = []
            for _ in range(num_tries):
                if timecheat:
                    next_time = model.get_time(state, last_t)
                else:
                    next_time = model.get_time(state)
                #print(f"next_time: {next_time}, next_state: {next_state}")
                current_pred.append(last_t + next_time)
            current_pred = torch.stack(current_pred)
            
            timestep = torch.Tensor([[timestep]])
            #print(torch.mean(current_pred))
            #loss = loss + mse_loss(next_time, timestep)
            score_loss = scoring_func(current_pred, timestep)
            if score_loss== 0:
                score_loss+=1e-15
            loss += (score_loss)# + thing should be useless

            state = model.get_new_state(state, timestep-last_t)
            last_t = timestep
        if iter % loss_print_interval == 0:
            progress_str = f"epoch: {iter} loss_sum: {loss :.4f}"
            print(progress_str)
            results.append((iter, loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()

        #for name, param in model.named_parameters():
        #    print(f"Parameter Name: {name}")
        #    print(f"Gradients: {param.grad}")
        #    print(f"Parameter Value: {param}")
        #    print(f"Parameter Shape: {param.shape}")
        #    print(f"Requires Gradient: {param.requires_grad}")
        #    print("-" * 40)
        #return
        if not (warmup_scheduler is None):
            with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()
        optimizer.zero_grad()

    progress_str = f"epoch: {iter} loss_sum: {loss :.4f}"
    print(progress_str)
    results.append((iter, loss))
    return results


def train_function_approx_multiple_variational(model, path_list, scoring_func,optimizer, state_size, 
        device, loss_func_kl, include_jump=False, logging_shift=0,
        lr_scheduler=None, warmup_period=None, warmup_scheduler=None,
        num_epochs=100, kl_weight=1, user_lr=.01, user_lr_decay=0.995,
        num_tries=20, timecheat=False, loss_print_interval=1):
    results =  []
    for iter in tqdm(range(-logging_shift, num_epochs-logging_shift)):
        
        loss_all = 0.
        loss_samples = 0.
        loss_kl = 0.
        #num_steps = 0
        random.shuffle(path_list)
        for datapoint_idx in range(len(path_list)):
            loss_samples_curr = 0.
            loss_kl_curr = 0.

            path, variational_means, variational_logvar, reaction_ratio, extras = path_list[datapoint_idx]
            variational_means, variational_logvar = torch.tensor(variational_means, 
                    requires_grad=True).to(device), torch.tensor(variational_logvar, requires_grad=True).to(device)
            
            last_t = 0
            variances = torch.exp(variational_logvar)
            state = variational_means + variances*torch.randn((1, state_size))

            for interaction_id in range(len(path)):
                timestep = path[interaction_id]
                current_pred = []
                for _ in range(num_tries):
                    if timecheat:
                        next_time = model.get_time(state, last_t)
                    else:
                        next_time = model.get_time(state)
                    #print(f"next_time: {next_time}, next_state: {next_state}")
                    current_pred.append(last_t + next_time)
                current_pred = torch.stack(current_pred)
                
                timestep = torch.Tensor([[timestep]])

                score_loss = scoring_func(current_pred, timestep)
                #if score_loss== 0:
                #    score_loss+=1e-15
                loss_samples_curr += (score_loss)# + thing should be useless

                state = model.get_new_state(state, timestep-last_t)
                if include_jump:
                    reactions_ratio_tensor = torch.tensor(reaction_ratio[interaction_id], dtype=torch.float32).view(1,-1)
                    state = model.jump(state, reactions_ratio_tensor)
                last_t = timestep
            
            loss_kl_curr = kl_weight * torch.sum(loss_func_kl(variational_means, variances))

            loss = loss_samples_curr + loss_kl_curr
            loss.backward()
            loss_all += loss.item()
            loss_samples += loss_samples_curr.item()
            loss_kl += loss_kl_curr.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
            optimizer.step()
            if not (warmup_scheduler is None):
                with warmup_scheduler.dampening():
                        if warmup_scheduler.last_step + 1 >= warmup_period:
                            lr_scheduler.step()
                #num_steps+=1
        
            # optimize variational
            torch.nn.utils.clip_grad_norm_(variational_means, max_norm=GRADIENT_CLIPPING_MAX_NORM)
            torch.nn.utils.clip_grad_norm_(variational_logvar, max_norm=GRADIENT_CLIPPING_MAX_NORM)
            with torch.no_grad():
                variational_means -= user_lr * variational_means.grad
                variational_logvar -= user_lr * variational_logvar.grad  
                #print(means)
                #means.grad.zero_()
                #logvar.grad.zero_()
                path_list[datapoint_idx][1] = variational_means.detach().tolist()
                path_list[datapoint_idx][2] = variational_logvar.detach().tolist()
        
            optimizer.zero_grad()
        #print(num_steps)
        if iter % loss_print_interval == 0:
            print(f"epoch: {iter+1+logging_shift} loss_sum_all: {loss_all :.4f}, loss_sum_freq: {loss_samples:.4f}, loss_sum_kl: {loss_kl:.4f}, lr: {lr_scheduler.get_lr()[0]:.7f}, userlr: {user_lr:.7f}")
            results.append((iter, loss_all, loss_samples, loss_kl))

            #for name, param in model.named_parameters():
            #    print(f"Parameter Name: {name}")
            #    print(f"Gradients: {param.grad}")
            #    print(f"Parameter Value: {param}")
            #    print(f"Parameter Shape: {param.shape}")
            #    print(f"Requires Gradient: {param.requires_grad}")
            #    print("-" * 40)
            #return
        user_lr *= user_lr_decay 

    print(f"epoch: {iter+1+logging_shift} loss_sum_all: {loss_all :.4f}, loss_sum_freq: {loss_samples:.4f}, loss_sum_kl: {loss_kl:.4f}, lr: {lr_scheduler.get_lr()[0]:.7f}, userlr: {user_lr:.7f}")
    results.append((iter, loss_all, loss_samples, loss_kl))
    return results


def train_density(model, dataloader, criterion, optimizer, warmup_scheduler, state_size, 
            lr_scheduler, warmup_period,
            num_epochs=100, loss_print_interval=1, print_grad=False):
    results = []
    for iter in tqdm(range(num_epochs)):
        loss_sum = 0.0
        
        for batch in dataloader:
            timesteps = batch['timestep'].unsqueeze(1)  # Add batch dimension
            frequencies = batch['frequency'].unsqueeze(1)
            state = torch.zeros((len(timesteps), state_size))
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(state, timesteps)
            
            loss = criterion(outputs, frequencies)  # Remove extra dimension from output
            loss.backward()
            loss_sum += loss.item()
            #print(f"loss: {loss} \tfrequencies: {frequencies} \n predicted: {outputs}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()    

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
        if iter % loss_print_interval == 0:
            print(f"epch: {iter} loss_sum: {loss_sum :.4f}")
            results.append((iter, loss_sum))
            if print_grad:
                for name, param in model.named_parameters():
                    print(f"Parameter Name: {name}")
                    print(f"Gradients: {param.grad}")
            #print(f"frequencies: {frequencies} \n predicted: {outputs}, time: {timesteps}")
            
    print(f"epch: {iter} loss_sum: {loss_sum :.4f}")
    results.append((iter, loss_sum))
    return results


def train_density_multiple_variational(model, dataloader_list, criterion, optimizer, warmup_scheduler, 
            state_size, device,
            lr_scheduler, warmup_period,loss_func_kl, kl_weight=1, user_lr=1,
            num_epochs=100, loss_print_interval=1, print_grad=False,
            train_bayesian_weight=0,logging_shift=0,
            user_lr_decay=1, state_consistancy_training=False, consistancy_weight=1):
    results = []
    mse_loss_func = nn.MSELoss()
    for iter in tqdm(range(num_epochs)):
        loss_sum_all = 0.0
        loss_sum_freq = 0.0
        loss_sum_kl = .0
        loss_state_consistancy = .0
        #num_updates = 0
        kl_loss_func_model = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        for i in range(-logging_shift,len(dataloader_list)-logging_shift):# not the nicest way to do things
            random.shuffle(dataloader_list)
            dataloader, variational_means, variational_logvar, extras = dataloader_list[i]
            for batch in dataloader:
                timesteps = batch['timestep'].unsqueeze(1)  # Add batch dimension
                frequencies = batch['frequency'].unsqueeze(1)
                size_curr = len(timesteps)
                variational_means, variational_logvar = torch.tensor(variational_means, 
                    requires_grad=True).to(device), torch.tensor(variational_logvar, requires_grad=True).to(device)
                variances = torch.exp(variational_logvar)
                state = variational_means + variances*torch.randn((size_curr, state_size))
                #curr_loss_state_consistancy = torch.tensor(0)
                curr_loss_state_consistancy = 0
                optimizer.zero_grad()
                
                # Forward pass
                outputs, new_state = model(state, timesteps, return_new_state=True)
                
                if state_consistancy_training:#extra loss to enforce state transition consistancy
                    mid_times = (.1 +0.8*torch.rand(size_curr, 1)) * timesteps
                    mid_states = model.evolve_state(state, mid_times)
                    rest_times = timesteps -mid_times
                    alternative_ends = model.evolve_state(mid_states, rest_times)
                    curr_loss_state_consistancy = consistancy_weight*mse_loss_func(new_state, alternative_ends)
                
                loss_freq = criterion(outputs, frequencies)  # Remove extra dimension from output

                curr_loss_kl = kl_weight * torch.sum(loss_func_kl(variational_means, variances))
                
                loss = loss_freq + curr_loss_kl + curr_loss_state_consistancy
                if train_bayesian_weight:
                    loss = loss + train_bayesian_weight * kl_loss_func_model(model)
                loss.backward()
                loss_sum_all += loss.item()
                loss_sum_freq += loss_freq.item()
                loss_sum_kl += curr_loss_kl.item()
                loss_state_consistancy += curr_loss_state_consistancy#.item()
                #print(f"loss: {loss} \tfrequencies: {frequencies} \n predicted: {outputs}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                optimizer.step()    

                # update vriational parameters
                torch.nn.utils.clip_grad_norm_(variational_means, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                torch.nn.utils.clip_grad_norm_(variational_logvar, max_norm=GRADIENT_CLIPPING_MAX_NORM)
                with torch.no_grad():
                    variational_means -= user_lr * variational_means.grad
                    variational_logvar -= user_lr * variational_logvar.grad  
                #print(means)
                #means.grad.zero_()
                #logvar.grad.zero_()
                dataloader_list[i][1] = variational_means.detach().tolist()
                dataloader_list[i][2] = variational_logvar.detach().tolist()
                
                with warmup_scheduler.dampening():
                    #num_updates+=1
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()
        
        if iter % loss_print_interval == 0:
            print(f"epoch: {iter+1+logging_shift} loss_sum_all: {loss_sum_all:.3f}, loss_sum_freq: {loss_sum_freq:.3f}, loss_sum_kl: {loss_sum_kl:.3f}, loss_state_consistancy: {loss_state_consistancy:.3f}, lr: {lr_scheduler.get_lr()[0]:.7f}, userlr: {user_lr:.7f}")
            results.append((iter, loss_sum_all, loss_sum_freq, loss_sum_kl, loss_state_consistancy))
            if print_grad:
                for name, param in model.named_parameters():
                    print(f"Parameter Name: {name}")
                    print(f"Gradients: {param.grad}")
            #print(f"num_updates: {num_updates}")
        user_lr *= user_lr_decay # lower variational lr,  fine for it to reach 0
        
    print(f"epoch: {iter+1+logging_shift} loss_sum_all: {loss_sum_all:.3f}, loss_sum_freq: {loss_sum_freq:.3f}, loss_sum_kl: {loss_sum_kl:.3f}, loss_state_consistancy: {loss_state_consistancy:.3f}, lr: {lr_scheduler.get_lr()[0]:.7f}, userlr: {user_lr:.7f}")
    results.append((iter, loss_sum_all, loss_sum_freq, loss_sum_kl, loss_state_consistancy))
            
    return results


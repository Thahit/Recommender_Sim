import torch
from functools import partial
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Dict, Callable
import math

class Base_Model(nn.Module):
    def __init__(self, ndims_in: int, ndims_out: int, model_hyp: Dict):
        super(Base_Model, self).__init__()

        last_w = ndims_in
        layers = []
        for w in model_hyp['layer_width']:
            layers.append(nn.Linear(last_w, w))
            layers.append(nn.SiLU())
            last_w = w
        layers.append(nn.Linear(last_w, ndims_out))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    #    for layer in self.model:
    #        x = layer(x)
    #    return x


#______________________________________state_____________________________-

class Ode_Function(Base_Model):
    def __init__(self, state_size: int, model_hyp: Dict):

        # time as input or integrate?, for now integrate
        super(Ode_Function, self).__init__(state_size, state_size, model_hyp)
    
    def forward(self, time, state):
        out = self.model(state)
        #out = torch.clip(out, min= -1e-2, max=1e2)
        return out


class User_State_Model(nn.Module):
    def __init__(self, state_size: int, model_hyp: Dict, noise: int = 0):
        super(User_State_Model, self).__init__()

        # time as input or integrate?, for now integrate
        #super(User_State_Model, self).__init__(state_size, state_size, model_hyp)

        # last layer is linear= no activation
        self.ode_func = Ode_Function(state_size, model_hyp)

        assert noise >= 0
        self.noise = noise
    
    def add_noise(self, state, h):
        if self.noise > 0:
            state += self.noise * torch.randn(state.shape) * h
        return state
    
    def forward(self, start_state: torch.Tensor, h: int, h_0: int = 0):
        t = torch.tensor([h_0, h], dtype=torch.float32)  # Time points for the integration
        
        # can specify max error if desired
        h_t = odeint(self.ode_func, start_state, t)  # h_t contains the states at t=0 and t=h
        end_state = h_t[1]

        return end_state

#______________________________________intensity_____________________________-
class User_State_Intensity_Model_ODE(Base_Model):
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Intensity_Model_ODE, self).__init__(state_size, 1, model_hyp)
    
    def forward(self, state):
        x = self.model(state)
        #relu  anything non-negative
        x = nn.functional.relu(x)
        return x

class User_State_Intensity_Model(nn.Module):
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Intensity_Model, self).__init__()
        self.ode = User_State_Intensity_Model_ODE(state_size, model_hyp)

    def odefunc(self, t, y, state, user_state_model):
        if t >1e-8:# otherwise there is errors
            state = user_state_model(state, t)        
        out = self.ode(state)
        return out

    def forward(self, time, state, state_model, h_0 = 0, interval_time=.01):
        '''
        evaluates the double integral in discrete points; interval_time determines the time steps.
        assume time only contains 1 element
        '''
        out = 0.
        num_intervals = int(math.ceil(time.item()/interval_time))
        #print(num_intervals)
        if num_intervals==0:
            print("time: ", time)
        for _ in range(num_intervals):
            state = state_model(start_state=state, h=interval_time)
            out += self.ode(state)
        out = nn.functional.relu(out)
        return out
    
    def forward_old(self, time, state, state_model, h_0 = 0):# too unstable to have a double integral
        t = torch.tensor([h_0, time], dtype=torch.float32)  # Time points for the integration
        initial_cond = torch.tensor([0.0]) 

        odefunc_partial = partial(self.odefunc, state=state, user_state_model=state_model)

        out = odeint(odefunc_partial, initial_cond, t)[1]
        out = nn.functional.relu(out)
        return out

class global_Intensity_Model_ODE(Base_Model):
    def __init__(self, time_size: int, model_hyp: Dict):

        super(global_Intensity_Model_ODE, self).__init__(time_size, 1, model_hyp)
    
    def forward(self, time, state):
        x = self.model(time.unsqueeze(0))
        #relu  anything non-negative
        #x = nn.functional.relu(x)
        return x

class global_Intensity_Model(nn.Module):
    def __init__(self, time_size: int, model_hyp: Dict):
        super(global_Intensity_Model, self).__init__()
        self.ode_func = global_Intensity_Model_ODE(time_size, model_hyp)

    def forward(self, time, h_0 = 0):
        t = torch.tensor([h_0, time], dtype=torch.float32)  # Time points for the integration
        initial_cond = torch.tensor([0.0]) # don't like that

        out = odeint(self.ode_func, initial_cond, t)[1]
        out = nn.functional.relu(out)
        return out

class Intensity_Model(nn.Module):
    def __init__(self, state_size: int, time_size: int, model_hyp: Dict):
        super(Intensity_Model, self).__init__()
        #self.intensity_models = []

        self.user_model = User_State_Intensity_Model(state_size, model_hyp["user_model_hyp"])
        self.global_model = global_Intensity_Model(time_size, model_hyp["global_model_hyp"])

    
    def forward(self, state, time, state_model):
        #maybe sshould input state model instead of state and convert everything into an ode
        intensity = 0
        intensity += self.user_model(state=state, time=time, state_model=state_model)

        # time might need to be incoded
        intensity += self.global_model(time)
        return intensity

#______________________________________interaction_____________________________-
class Single_Interaction_Model(Base_Model):
    def __init__(self, state_size: int, recommendation_size: int, num_outcomes: int, model_hyp: Dict):

        super(Single_Interaction_Model, self).__init__(state_size + recommendation_size, num_outcomes, model_hyp)
    
    def forward(self, state, recommendation):        
        x = torch.cat((state, recommendation), dim=1)
        x = self.model(x)

        x = nn.functional.relu(x)
        return x


class Interaction_Model(nn.Module):
    def __init__(self, state_size: int, recommendation_size: int, num_outcomes: int, interaction_model_hyp: Dict):
        super(Interaction_Model, self).__init__()

        self.single_interaction_model = Single_Interaction_Model(state_size, recommendation_size, num_outcomes, interaction_model_hyp)

    def forward(self, state, recommendations):
        out = []
        #for i in range(recommendations.size(1)):# iter over recommendations
        for i in range(len(recommendations)):
            recommendation = recommendations[i].view(1,1)
            curr_out = self.single_interaction_model(state, recommendation)
            out.append(curr_out)
        out = torch.stack(out, dim=1)
        return out


#______________________________________jump_____________________________-
class Jump_Model(Base_Model):
    '''
    might depend on recommennded items
    might depend on outcomes
        might depend only on summary stats of outcomes
    ...
    '''
    def __init__(self, state_size: int, num_outcomes: int, model_hyp: Dict):
        super(Jump_Model, self).__init__(state_size + num_outcomes, state_size, model_hyp)

        
    def encode_outcomes(self, recommendation_outcomes):
        return torch.sum(recommendation_outcomes, dim=0)
    
    def forward(self, state, recommendation_outcomes):
        # for now just use recommendations a sinput? caant since different sizes-> need summary stats

        #recommendation_outcomes = recommendation_outcomes.view(recommendation_outcomes.size(0), -1)
        recommendation_outcomes = self.encode_outcomes(recommendation_outcomes).view(1,-1)
        x = torch.cat((state, recommendation_outcomes), dim=1)
        x = self.model(x)# no activation needed?

        return x # jump_size

#_______________________________________combined model______________________________-
class User_simmulation_Model(nn.Module):
    def __init__(self, hyperparameter_dict: Dict):
        super(User_simmulation_Model, self).__init__()
        self.state_size = hyperparameter_dict["state_size"]
        self.time_size = 1# might change if we encode time
        self.recommendation_size = hyperparameter_dict["recom_dim"]
        self.num_outcomes = hyperparameter_dict["num_recom"]# different reactions to outcomes
        self.num_interaction_outcomes = hyperparameter_dict["num_interaction_outcomes"]
        #init all sub-modules

        self.state_model = User_State_Model(self.state_size, 
                                            hyperparameter_dict["state_model"]["model_hyp"],
                                            noise=hyperparameter_dict["state_model"].get("noise", 0))

        self.intensity_model = Intensity_Model(self.state_size, time_size=self.time_size,
                                               model_hyp=hyperparameter_dict["intensity_model"]["model_hyp"])

        self.interaction_model = Interaction_Model(self.state_size, self.recommendation_size,
                                                   self.num_outcomes, hyperparameter_dict["interaction_model"]["model_hyp"])
        
        jump_model_interaction_size = self.num_outcomes * self.num_interaction_outcomes# could change if weuse summary stats
        jump_model_interaction_size = self.num_outcomes
        self.jump_model = Jump_Model(self.state_size, jump_model_interaction_size, hyperparameter_dict["jump_model"]["model_hyp"])

    def forward(self, state, recommendations):
        raise NotImplementedError
    
    def init_state(self, state):
        self.state = state
    
    def eval_intensity(self, h):
        return self.intensity_model(self.state, h, state_model=self.state_model)

    def evolve_state(self, h):
        ''' 
        return state after time interval h
        Returns:
            tensor: state
        '''
        #h = self.encode_time(h)
        self.state = self.state_model(self.state, h)
    
    def view_recommendations(self, recommendations):
        return self.interaction_model(self.state, recommendations)

    def jump(self, interactions):
        self.state += self.jump_model(self.state, interactions)
    
    def get_user_state(self):
        return self.state

if __name__ == "__main__":
    # test code

    state_size = 2
    num_interaction_outcomes = 2
    time_size = 1
    num_recomm = 2
    recom_dim=1

    # state model
    user_state_dict = {"model_hyp": {"layer_width": [3,3]}}
    state_model = User_State_Model(state_size=state_size, model_hyp=user_state_dict["model_hyp"])
    state = torch.randn((state_size))
    state = state.unsqueeze(0)
    target_time = 1.
    new_state = state_model(state, target_time)
    print("new_state: ", new_state, "\t shape: ", new_state.shape)

    # intensity model
    intensity_state_dict = {"model_hyp": {"user_model_hyp": {"layer_width": [3,3]},
                                          "global_model_hyp": {"layer_width": [3,3]}}
                            }
    time_tensor = torch.as_tensor([target_time])
    intensity_model = Intensity_Model(state_size=state_size, time_size= time_size, model_hyp= intensity_state_dict["model_hyp"])
    intensity = intensity_model(new_state, time_tensor)
    print("intensity: ", intensity, "\t shape: ", intensity.shape)

    # interaction model
    interaction_state_dict = {"model_hyp": {"layer_width": [3,3]}
                            }

    recommendations = torch.randn((1,num_recomm, recom_dim))
    #recommendations = recommendations.unsqueeze(0)
    intensity_model = Interaction_Model(state_size=state_size, recommendation_size=recom_dim,
                                        num_outcomes=num_interaction_outcomes, interaction_model_hyp=interaction_state_dict["model_hyp"])
    interactions = intensity_model(new_state, recommendations)
    print("Interactions: ", interactions, "\t shape: ", interactions.shape)
    
    # backprop?
    true_interactions = torch.randn((1,num_recomm, num_interaction_outcomes))
    loss = nn.functional.mse_loss(interactions, true_interactions)
    loss.backward()

    # jump
    jump_state_dict = {"model_hyp": {"layer_width": [3,3]}
                        }

    jump_model = Jump_Model(state_size, num_interaction_outcomes*num_recomm, jump_state_dict["model_hyp"])
    jump = jump_model(new_state, interactions)
    print("jump: ", jump, "\t shape: ", jump.shape)
    new_state += jump
    print("new_state: ", new_state, "\t shape: ", new_state.shape)

    #combined_model
    hyperparameter_dict = {"state_size": state_size, "state_model": user_state_dict, "num_interaction_outcomes": num_interaction_outcomes,
                           "intensity_model": intensity_state_dict, "num_recom" : num_recomm,
                            "recom_dim":recom_dim, "interaction_model": interaction_state_dict,
                            "jump_model": jump_state_dict}
    combined_model = User_simmulation_Model(hyperparameter_dict)
    combined_model.init_state(state)
    combined_model.evolve_state(time_tensor)
    interactions = combined_model.view_recommendations(recommendations)
    print("Interactions: ", interactions, "\t shape: ", interactions.shape)
    combined_model.jump(interactions)


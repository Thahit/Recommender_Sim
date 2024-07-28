import torch
from functools import partial
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Dict, Callable
import math

ODE_GRADIENT_CLIP = 1e4
MIN_INTEGRAL = 1e-2
EPSILON = 1e-6 # numerical errors
class Base_Model(nn.Module):
    """
    A base neural network model with customizable layer widths.

    Args:
        ndims_in (int): Number of input dimensions.
        ndims_out (int): Number of output dimensions.
        model
    """
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
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
    #    for layer in self.model:
    #        x = layer(x)
    #    return x


#______________________________________state_____________________________-

class Ode_Function(Base_Model):
    """
    A neural network model specifically for ODE functions, inheriting from Base_Model.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        # time as input or integrate?, for now integrate
        super(Ode_Function, self).__init__(state_size, state_size, model_hyp)
    
    def forward(self, time, state):
        """
        Forward pass for ODE functions.

        Args:
            time (torch.Tensor): Time tensor.
            state (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Output tensor representing the derivative of the state.
        """
        out = self.model(state)
        out = torch.clip(out, min= -ODE_GRADIENT_CLIP, max=ODE_GRADIENT_CLIP)
        return out
    

class Ode_Function_Conditioned(Base_Model):
    """
    A neural network model specifically for ODE functions, inheriting from Base_Model.

    Args:
        state_size (int): Size of the state vector.
        user_params_size (int): Size of the user parameter vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, user_params_size: int, model_hyp: Dict):
        # time as input or integrate?, for now integrate
        super(Ode_Function, self).__init__(state_size + user_params_size, state_size, model_hyp)
    
    def forward(self, time, state, user_params):
        """
        Forward pass for ODE functions.

        Args:
            time (torch.Tensor): Time tensor.
            state (torch.Tensor): State tensor.
            user_params (torch.Tensor): User Parameter tensor.

        Returns:
            torch.Tensor: Output tensor representing the derivative of the state.
        """
        out = torch.cat(state, user_params, dim=1)
        out = self.model(out)
        out = torch.clip(out, min= -ODE_GRADIENT_CLIP, max=ODE_GRADIENT_CLIP)
        return out


class User_State_Model(nn.Module):
    """
    A model for simulating state evolution using ODEs, with optional noise addition.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters for the ODE function.
        noise (int): scaling of the noise added.
    """
    def __init__(self, state_size: int, model_hyp: Dict, noise):
        super(User_State_Model, self).__init__()

        # time as input or integrate?, for now integrate
        #super(User_State_Model, self).__init__(state_size, state_size, model_hyp)

        # last layer is linear= no activation
        self.ode_func = Ode_Function(state_size, model_hyp)

        self.noise = noise
        assert self.noise >= 0
    
    def add_noise(self, state, h):
        """
        Add noise to the state.

        Args:
            state (torch.Tensor): State tensor.
            h (int): Time step size.

        Returns:
            torch.Tensor: Noisy state tensor.
        """
        if self.noise > 0:
            state += self.noise * torch.randn(state.shape) * h
        return state
    
    def forward_old(self, state: torch.Tensor, h: int, h_0: int = 0):# ode solver
        """
        Evolve the state from time h_0 to h using ODE integration.

        Args:
            state (torch.Tensor): Initial state tensor.
            h (int): Time interval for evolution.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: State tensor at time h.
        """
        
        # can specify max error if desired
        if h > MIN_INTEGRAL:
            t = torch.tensor([h_0, h], dtype=torch.float32)  # Time points for the integration
            state = odeint(self.ode_func, state, t)[1]  # h_t contains the states at t=0 and t=h

        state = self.add_noise(state, h)
        return state
    
    def forward(self, state: torch.Tensor, h: int, interval_time=.02):# discrete
        """
        Evaluate the integral of the state function over discrete time intervals.

        Args:
            state (torch.Tensor): State tensor.
            h (float): Time. 
            interval_time (float): Time step size. Default is 0.02.

        Returns:
            torch.Tensor: state tensor.
        """
        num_intervals = int(math.floor(h/interval_time))

        for _ in range(num_intervals):
            state = state+self.ode_func(interval_time, state)*interval_time
        
        rest_interval = h%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            state = state+self.ode_func(rest_interval, state)

        #state = self.add_noise(state, h)
        return state


class Conditioned_User_State_Model(nn.Module):
    """
    A model for simulating state evolution using ODEs, with optional noise addition.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters for the ODE function.
        noise (int): scaling of the noise added.
    """
    def __init__(self, state_size: int, user_params_size: int, model_hyp: Dict, noise):
        super(Conditioned_User_State_Model, self).__init__()

        # not sure if user params are drawn or if we use the means/variance, I take the drawn ones for now
        self.ode_func = Ode_Function_Conditioned(state_size, user_params_size, model_hyp)

        self.noise = noise
        assert self.noise >= 0
    
    def add_noise(self, state, h):
        """
        Add noise to the state.

        Args:
            state (torch.Tensor): State tensor.
            h (int): Time step size.

        Returns:
            torch.Tensor: Noisy state tensor.
        """
        if self.noise > 0:
            state += self.noise * torch.randn(state.shape) * h
        return state
    
    def forward(self, start_state: torch.Tensor, user_params: torch.Tensor, h: int, h_0: int = 0):
        """
        Evolve the state from time h_0 to h using ODE integration.

        Args:
            start_state (torch.Tensor): Initial state tensor.
            h (int): Time interval for evolution.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: State tensor at time h.
        """
        t = torch.tensor([h_0, h], dtype=torch.float32)  # Time points for the integration
        
        odefunc_partial = partial(self.ode_func, user_params=user_params)

        h_t = odeint(odefunc_partial, start_state, t)  # h_t contains the states at t=0 and t=h
        end_state = h_t[1]

        return end_state

class User_State_Model_ensemble(nn.Module):
    """
    A model for simulating state evolution using ODEs, with optional noise addition.
    Is an ensemble of User_state_models

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters for the ODE function.
        noise (int): scaling of the noise added.
        num_models (int): size of ensemble.
    """
    def __init__(self, state_size: int, model_hyp: Dict, noise, num_models: int = 10, return_var = False):
        super(User_State_Model_ensemble, self).__init__()
        # for now not very bayesian but a skeleton

        # last layer is linear= no activation
        self.ensemble = []
        self.return_var = return_var
        for _ in range(num_models):
            self.ensemble.append(User_State_Model(state_size=state_size, model_hyp=model_hyp, noise=noise))
    
    def forward(self, start_state: torch.Tensor, h: int, h_0: int = 0):
        """
        Evolve the state from time h_0 to h using ensemble_members

        Args:
            start_state (torch.Tensor): Initial state tensor.
            h (int): Time interval for evolution.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: State tensor at time h.
        """
        outputs = []
        for member in self.ensemble:
            outputs.append(member(start_state, h))

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        if not self.return_var:
            return mean
        uncertainty = outputs.var(dim=0)
        return mean, uncertainty
#______________________________________intensity_____________________________-
class User_State_Intensity_Model_ODE(Base_Model):
    """
    A neural network model for computing the intensity function of state changes in an ODE.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Intensity_Model_ODE, self).__init__(state_size, 1, model_hyp)
    
    def forward(self, state):
        """
        Compute the intensity based on the state.

        Args:
            state (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        x = self.model(state)
        #relu  anything non-negative
        x = nn.functional.relu(x)
        return x

class User_State_Intensity_Model(nn.Module):
    """
    A model for evaluating intensity functions with ODE integration over time.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Intensity_Model, self).__init__()
        self.ode = User_State_Intensity_Model_ODE(state_size, model_hyp)

    def odefunc(self, t, y, state, user_state_model):
        """
        ODE function used for integration.

        Args:
            t (float): Current time.
            y (torch.Tensor): State tensor.
            state (torch.Tensor): State tensor.
            user_state_model (Callable): Function for updating the state.

        Returns:
            torch.Tensor: Derivative of the state.
        """
        #if t >1e-8:# otherwise there is errors
        state = user_state_model(state, t)        
        out = self.ode(state)
        return out

    def forward(self, time, state, state_model, h_0 = 0, interval_time=.1):
        """
        Evaluate the integral of the intensity function over discrete time intervals.

        Args:
            time (torch.Tensor): Time tensor. (assume only 1 element)
            state (torch.Tensor): State tensor.
            state_model (Callable): Function for evolving the state.
            h_0 (int): Initial time. Default is 0.
            interval_time (float): Time step size. Default is 0.05.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        out = 0.
        num_intervals = int(math.floor(time.item()/interval_time))

        for _ in range(num_intervals):
            state = state_model(state, h=interval_time)
            out += self.ode(state) *interval_time
        
        rest_interval = time.item()%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            state = state_model(state, h=rest_interval)
            out += self.ode(state) *rest_interval

        #needs to be >= 0   maybe also <=1?
        #out = nn.functional.relu(out)
        out = nn.functional.sigmoid(out)
        return out
    
    def forward_old(self, time, state, state_model, h_0 = 0):# too unstable to have a double integral
        """
        Evaluate the integral of the intensity function using ODE integration.

        Args:
            time (torch.Tensor): Time tensor.
            state (torch.Tensor): State tensor.
            state_model (Callable): Function for evolving the state.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        t = torch.tensor([h_0, time], dtype=torch.float32)  # Time points for the integration
        initial_cond = torch.tensor([0.0]) 

        odefunc_partial = partial(self.odefunc, state=state, user_state_model=state_model)

        out = odeint(odefunc_partial, initial_cond, t)[1]
        out = nn.functional.relu(out)
        return out

class global_Intensity_Model_ODE(Base_Model):
    """
    A neural network model for computing the global intensity function (constant for all users) of time.

    Args:
        time_size (int): Size of the time vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, time_size: int, model_hyp: Dict):

        super(global_Intensity_Model_ODE, self).__init__(time_size, 1, model_hyp)
    
    def forward(self, time, intensity):
        """
        Compute the global intensity based on time.

        Args:
            time (torch.Tensor): Time tensor.
            intensity (torch.Tensor): intensity tensor.

        Returns:
            torch.Tensor: Global intensity tensor.
        """
        x = self.model(time.unsqueeze(0))
        #relu  anything non-negative
        #x = nn.functional.relu(x)
        return x

class global_Intensity_Model(nn.Module):
    """
    A model for computing the global intensity function with ODE integration.

    Args:
        time_size (int): Size of the time vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, time_size: int, model_hyp: Dict):
        super(global_Intensity_Model, self).__init__()
        self.ode_func = global_Intensity_Model_ODE(time_size, model_hyp)

    def forward_old(self, time, h_0 = 0):
        """
        Compute the global intensity based on time using ODE integration.

        Args:
            time (torch.Tensor): Time tensor.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: Global intensity tensor.
        """
        t = torch.tensor([h_0, time], dtype=torch.float32)  # Time points for the integration
        initial_cond = torch.tensor([0.0]) # don't like that

        out = odeint(self.ode_func, initial_cond, t)[1]
        out = nn.functional.sigmoid(out)
        return out
    
    def forward(self, time, h_0 = 0, interval_time=.05):
        """
        Evaluate the integral of the intensity function over discrete time intervals.

        Args:
            time (torch.Tensor): Time tensor. (assume only 1 element)
            h_0 (int): Initial time. Default is 0.
            interval_time (float): Time step size. Default is 0.02.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        out = torch.tensor([0.0])
        num_intervals = int(math.floor(time.item()/interval_time))

        for _ in range(num_intervals):
            out = out + self.ode_func(time, out) *interval_time
        
        rest_interval = time.item()%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            out = out + self.ode_func(time, out) *rest_interval

        #needs to be >= 0   maybe also <=1?
        #out = nn.functional.relu(out)
        out = nn.functional.sigmoid(out)
        return out

class Intensity_Model(nn.Module):
    """
    A combined model for computing intensity functions from user state and global time.

    Args:
        state_size (int): Size of the state vector.
        time_size (int): Size of the time vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, time_size: int, model_hyp: Dict):
        super(Intensity_Model, self).__init__()
        #self.intensity_models = []

        self.user_model = User_State_Intensity_Model(state_size, model_hyp["user_model_hyp"])
        self.global_model = global_Intensity_Model(time_size, model_hyp["global_model_hyp"])

    
    def forward(self, state, time_delta, time, state_model):
        """
        Compute the total intensity as the sum of user and global intensity models.

        Args:
            state (torch.Tensor): State tensor.
            time (torch.Tensor): Time tensor.
            state_model (Callable): Function for evolving the state.

        Returns:
            torch.Tensor: Total intensity tensor.
        """
        intensity = 0
        user_intensity = self.user_model(state=state, time=time_delta, state_model=state_model)
        intensity += user_intensity
        # time might need to be incoded
        global_intensity = self.global_model(time)
        intensity += global_intensity
        intensity = torch.clamp(intensity, min=EPSILON, max=1-EPSILON)
        return intensity, user_intensity, global_intensity

#______________________________________interaction_____________________________-
class Single_Interaction_Model(Base_Model):
    """
    A model for computing interaction outcomes based on state and recommendations.

    Args:
        state_size (int): Size of the state vector.
        recommendation_size (int): Size of the recommendation vector.
        num_outcomes (int): Number of outcome dimensions.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, recommendation_size: int, num_outcomes: int, model_hyp: Dict):

        super(Single_Interaction_Model, self).__init__(state_size + recommendation_size, num_outcomes, model_hyp)
    
    def forward(self, state, recommendation):  
        """
        Compute interaction outcomes based on state and a single recommendation.

        Args:
            state (torch.Tensor): State tensor.
            recommendation (torch.Tensor): Recommendation tensor.

        Returns:
            torch.Tensor: Interaction outcomes tensor.
        """      
        x = torch.cat((state, recommendation), dim=1)
        x = self.model(x)

        x = nn.functional.relu(x)
        return x


class Interaction_Model(nn.Module):
    """
    A model for computing interactions for a batch of recommendations.

    Args:
        state_size (int): Size of the state vector.
        recommendation_size (int): Size of the recommendation vector.
        num_outcomes (int): Number of outcome dimensions.
        interaction_model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, recommendation_size: int, num_outcomes: int, interaction_model_hyp: Dict):
        super(Interaction_Model, self).__init__()

        self.single_interaction_model = Single_Interaction_Model(state_size, recommendation_size, num_outcomes, interaction_model_hyp)

    def forward(self, state, recommendations):
        """
        Compute interaction outcomes for a batch of recommendations.

        Args:
            state (torch.Tensor): State tensor.
            recommendations (torch.Tensor): Batch of recommendation tensors.

        Returns:
            torch.Tensor: Interaction outcomes tensor for each recommendation.
        """
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
    """
    A model for computing jump sizes based on state and recommendation outcomes.

    Args:
        state_size (int): Size of the state vector.
        num_outcomes (int): Number of outcome dimensions.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, num_outcomes: int, model_hyp: Dict):
        super(Jump_Model, self).__init__(state_size + num_outcomes, state_size, model_hyp)

        
    def encode_outcomes(self, recommendation_outcomes):
        """
        Encode recommendation outcomes into a summary statistic.

        Args:
            recommendation_outcomes (torch.Tensor): Tensor of recommendation outcomes.

        Returns:
            torch.Tensor: Encoded outcomes tensor.
        """
        return torch.sum(recommendation_outcomes, dim=0)
    
    def forward(self, state, recommendation_outcomes):
        """
        Compute the jump size based on state and recommendation outcomes.

        Args:
            state (torch.Tensor): State tensor.
            recommendation_outcomes (torch.Tensor): Recommendation outcomes tensor.

        Returns:
            torch.Tensor: Jump size tensor.
        """
        #recommendation_outcomes = recommendation_outcomes.view(recommendation_outcomes.size(0), -1)
        recommendation_outcomes = self.encode_outcomes(recommendation_outcomes).view(1,-1)
        x = torch.cat((state, recommendation_outcomes), dim=1)
        x = self.model(x)# no activation needed?

        return x # jump_size

#_______________________________________combined model______________________________-
class User_simmulation_Model(nn.Module):
    """
    A combined model for simulating user behavior, including state evolution, intensity evaluation, interactions, and jumps.

    Args:
        hyperparameter_dict (Dict): Dictionary containing all model hyperparameters, including 'state_size', 'time_size', 'recom_dim', 'num_recom', 'num_interaction_outcomes', and sub-model hyperparameters.
    """
    def __init__(self, hyperparameter_dict: Dict):
        super(User_simmulation_Model, self).__init__()
        self.state_size = hyperparameter_dict["state_size"]
        self.time_size = 1# might change if we encode time
        self.recommendation_size = hyperparameter_dict["recom_dim"]
        #self.num_recom = hyperparameter_dict["num_recom"]# different reactions to outcomes
        self.num_interaction_outcomes = hyperparameter_dict["num_interaction_outcomes"]
        #init all sub-modules

        self.state_model = User_State_Model(self.state_size, 
                                            hyperparameter_dict["state_model"]["model_hyp"],
                                            noise=hyperparameter_dict["state_model"].get("noise", 0))

        self.intensity_model = Intensity_Model(self.state_size, time_size=self.time_size,
                                               model_hyp=hyperparameter_dict["intensity_model"]["model_hyp"])

        self.interaction_model = Interaction_Model(self.state_size, self.recommendation_size,
                                                   self.num_interaction_outcomes, hyperparameter_dict["interaction_model"]["model_hyp"])
        
        jump_model_interaction_size = self.num_interaction_outcomes * self.num_interaction_outcomes# could change if weuse summary stats
        jump_model_interaction_size = self.num_interaction_outcomes
        self.jump_model = Jump_Model(self.state_size, jump_model_interaction_size, hyperparameter_dict["jump_model"]["model_hyp"])

    def forward(self, state, recommendations):
        raise NotImplementedError
    
    def init_state(self, state):
        """
        Initialize the state of the model.

        Args:
            state (torch.Tensor): Initial state tensor.
        """
        self.state = state
    
    def eval_intensity(self, time_delta, time, return_all: bool = False):
        """
        Evaluate the intensity function at a given time delta.

        Args:
            time_delta (torch.Tensor): Time delta tensor.
            time(torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        overall_intensity, user_intensity, local_intensity = self.intensity_model(self.state, time_delta, time, state_model=self.state_model)
        if return_all:
            return overall_intensity, user_intensity, local_intensity 
        return overall_intensity

    def evolve_state(self, h):
        """
        Evolve the state of the model over a time interval.

        Args:
            h (torch.Tensor): Time interval.
        """
        #h = self.encode_time(h)
        self.state = self.state_model(self.state, h)
    
    def view_recommendations(self, recommendations):
        """
        View the interaction outcomes based on the current state and recommendations.

        Args:
            recommendations (torch.Tensor): Batch of recommendation tensors.

        Returns:
            torch.Tensor: Interaction outcomes tensor.
        """
        return self.interaction_model(self.state, recommendations)

    def jump(self, interactions):
        """
        Apply a jump to the state based on interactions.

        Args:
            interactions (torch.Tensor): Interaction outcomes tensor.
        """
        self.state += self.jump_model(self.state, interactions)
    
    def get_user_state(self):
        """
        Get the current state of the user.

        Returns:
            torch.Tensor: Current state tensor.
        """
        return self.state

class Conditioned_User_simmulation_Model(User_simmulation_Model):
    """
    A combined model for simulating user behavior, including state evolution, intensity evaluation, interactions, and jumps.

    Args:
        hyperparameter_dict (Dict): Dictionary containing all model hyperparameters, including 'state_size', 'time_size', 'recom_dim', 'num_recom', 'num_interaction_outcomes', and sub-model hyperparameters.
    """
    def __init__(self, hyperparameter_dict: Dict):
        super(Conditioned_User_simmulation_Model, self).__init__()

        self.state_model = Conditioned_User_simmulation_Model(self.state_size, hyperparameter_dict["user_params_size"],
                                            hyperparameter_dict["state_model"]["model_hyp"],
                                            noise=hyperparameter_dict["state_model"].get("noise", 0))

    
    def init_state(self, state, user_params):
        """
        Initialize the state of the model.

        Args:
            state (torch.Tensor): Initial state tensor.
        """
        self.state = state
        self.user_params = user_params
    

    def evolve_state(self, h):
        """
        Evolve the state of the model over a time interval.

        Args:
            h (torch.Tensor): Time interval.
        """
        #h = self.encode_time(h)
        self.state = self.state_model(self.state, user_params=self.user_params,
                                      h=h)

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


import torch
from functools import partial
import torch.nn as nn
from torchdiffeq import odeint
from typing import Dict
import math
import numpy as np
import torchbnn as bnn

ODE_GRADIENT_CLIP = 1e4
MIN_INTEGRAL = 1e-5
EPSILON = 1e-20 # numerical errors


def smoothclamp_0_1(x):
    x= torch.as_tensor(x)
    x = nn.functional.tanh(2*x - 1)+1
    x = x/2
    return x


def smoothclamp(x):
    #x = nn.functional.softplus(x)
    #return x
    #x= torch.as_tensor(x)
    x = nn.functional.tanh(x/35)+1
    x = x*35
    return x


class SignWaveEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_freq=1):
        """
        Initialize the SignWaveEmbedding module.
        
        Args:
            embedding_dim (int): Dimension of the resulting embedding vector.
            max_freq (int): Maximal frequency used.

        """
        assert embedding_dim %2==0
        super(SignWaveEmbedding, self).__init__()
        self.num_frequencies = embedding_dim//2
        self.embedding_dim = embedding_dim

        # Create frequency values. Use logarithmic spacing to cover a range of scales.
        self.frequencies = torch.logspace(-4, np.log10(max_freq), self.num_frequencies, base=2, dtype=torch.float32)

    def forward(self, x):
        """
        Generate the sign wave embeddings for the input scalar.
        
        Args:
            x (torch.Tensor): Input tensor of scalars with shape (batch_size,).
        
        Returns:
            torch.Tensor: Embedding tensor with shape (batch_size, embedding_dim).
        """
        #x = x.unsqueeze(1)  # Add dimension for broadcasting

        # Compute the embedding vector
        sin_embeddings = torch.sin(x * self.frequencies)
        cos_embeddings = torch.cos(x * self.frequencies)

        # Concatenate sine and cosine embeddings
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=-1)

        return embeddings


#_________________________________________base_____________________________
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
            
        if model_hyp.get("bayesian", False):
            print("bayesian")
            for w in model_hyp['layer_width']:
                layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=.1, 
                        in_features=last_w, out_features=w))
                layers.append(nn.SiLU())
                last_w = w
            layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=.1, 
                        in_features=last_w, out_features=ndims_out))
        else:
            for w in model_hyp['layer_width']:
                layers.append(nn.Linear(last_w, w))
                layers.append(nn.SiLU())
                last_w = w
            layers.append(nn.Linear(last_w, ndims_out))

        #self._initialize_weights()
        self.model = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights and biases with normal distribution (mean=0, std=1).
        """
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0, std=1)

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

class Ode_Function_linear(nn.Module):# useless
    """
    A neural network model specifically for ODE functions, inheriting from Base_Model.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters. Ignored
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(Ode_Function_linear, self).__init__()
        self.model = nn.Linear(state_size, state_size)
    
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
        
        return out


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
        #out = torch.clip(out, min= -ODE_GRADIENT_CLIP, max=ODE_GRADIENT_CLIP)
        #out = smoothclamp(out, -ODE_GRADIENT_CLIP, ODE_GRADIENT_CLIP)
        
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
        super(Ode_Function_Conditioned, self).__init__(state_size + user_params_size, 
                                state_size, model_hyp)
    
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
        out = torch.cat((state, user_params), dim=1)
        out = self.model(out)
        #out = torch.clip(out, min= -ODE_GRADIENT_CLIP, max=ODE_GRADIENT_CLIP)
        #out = smoothclamp(out, -ODE_GRADIENT_CLIP, ODE_GRADIENT_CLIP)
        return out


class User_State_Model(nn.Module):
    """
    A model for simulating state evolution using ODEs, with optional noise addition.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters for the ODE function.
        noise (int): scaling of the noise added.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Model, self).__init__()

        # last layer is linear= no activation
        self.ode_func = Ode_Function(state_size, model_hyp)

        self.noise = model_hyp.get("noise", 0)
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
            state =  state + (self.noise * torch.randn(state.shape) * h).clone()
        return state
    
    '''def forward(self, state: torch.Tensor, h: int, h_0: int = 0):# ode solver
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
        return state'''
    
    def forward(self, state: torch.Tensor, h: int, interval_time=.1):# discrete
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
            state = state+(self.ode_func(interval_time, state)*interval_time).clone()
        
        rest_interval = h%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            state = state + (self.ode_func(rest_interval, state)*rest_interval).clone()

        state = self.add_noise(state, h)
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
            state = state + (self.noise * torch.randn(state.shape) * h).clone()
        return state
    
    '''
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
    '''

    def forward(self, state: torch.Tensor, user_params: torch.Tensor, h: int, interval_time=.05):# discrete
        """
        Evaluate the integral of the state function over discrete time intervals.

        Args:
            state (torch.Tensor): State tensor.
            user_params (torch.Tensor): User parameters to condition on.
            h (float): Time. 
            interval_time (float): Time step size. Default is 0.02.

        Returns:
            torch.Tensor: state tensor.
        """
        num_intervals = int(math.floor(h/interval_time))
        odefunc_partial = partial(self.ode_func, user_params=user_params)

        for _ in range(num_intervals):
            state = state + (odefunc_partial(interval_time, state)*interval_time).clone()
        
        rest_interval = h%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            state = state + (odefunc_partial(rest_interval, state)).clone()

        #state = self.add_noise(state, h)
        return state


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
    
    def forward(self, y, state):
        """
        Compute the intensity based on the state.

        Args:
            y (torch.Tensor): intensity.
            state (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        #x = torch.cat((state, y), dim=1)
        x = state
        x = self.model(x)
        x = nn.functional.softplus(x)
        
        return x


class User_State_Intensity_Model_simple(Base_Model):
    """
    A neural network model for computing the intensity function of state changes in an ODE.

    Args:
        state_size (int): Size of the state vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(User_State_Intensity_Model_simple, self).__init__(state_size, 1, model_hyp)
    
    def forward(self,  state):# maybe can depend on time too
        """
        Compute the intensity based on the state.

        Args:
            state (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        #x = torch.cat((state, t), dim=1)
        x = self.model(state)
        x = nn.functional.softplus(x)
        #x = nn.functional.relu(x)
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
        self.ode = User_State_Intensity_Model_ODE(state_size, model_hyp)# 1 for intensity

    def odefunc(self, t, y, state, user_state_model):
        """
        ODE function used for integration.

        Args:
            t (float): Current time.
            y (torch.Tensor): initial_val.
            state (torch.Tensor): State tensor.
            user_state_model (Callable): Function for updating the state.

        Returns:
            torch.Tensor: Derivative of the state.
        """
        #if t >1e-8:# otherwise there is errors
        updated_state = user_state_model(state= state, h=t)        
        out = self.ode(y, updated_state)

        return out
    
    '''
    def forward(self, time, state, state_model, h_0 = 0, interval_time=.05, user_params=None):
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
        out = torch.zeros((1, 1), requires_grad=True)

        num_intervals = int(math.floor(time.item()/interval_time))

        if not(user_params is None):
            state_model = partial(state_model, user_params=user_params)

        for _ in range(num_intervals):
            #state = state_model(state, h=interval_time)
            #out = out + (self.ode(y, state) *interval_time).clone()
            out = out + (self.odefunc(interval_time, out, state, state_model) * interval_time).clone()
        
        rest_interval = time.item()%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            #state = state_model(state, h=rest_interval)
            #out = out + (self.ode(y, state) *rest_interval).clone()
            out = out + (self.odefunc(interval_time, out, state, state_model) * rest_interval).clone()

        #needs to be >= 0   maybe also <= 1?
        #out = nn.functional.relu(out)
        #out = nn.functional.sigmoid(out)
        #out = torch.exp(-out)
        return out
    '''

    '''
    def forward(self, time, state, state_model, h_0 = 0):# too unstable to have a double integral
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
        initial_cond = torch.zeros((1, 1), requires_grad=True)

        odefunc_partial = partial(self.odefunc, state=state, user_state_model=state_model)

        out = odeint(odefunc_partial, initial_cond, t)[1]
        #out = nn.functional.relu(out)
        out = torch.exp(-out).clone()
        return out
    '''

    def forward(self, state, h, intensity):
        return self.ode(intensity, state)


class global_Intensity_Model_ODE(Base_Model):
    """
    A neural network model for computing the global intensity function (constant for all users) of time.

    Args:
        time_size (int): Size of the time vector.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, time_size: int, model_hyp: Dict):

        super(global_Intensity_Model_ODE, self).__init__(time_size, 1, model_hyp)
    
    def forward(self, time):
        """
        Compute the global intensity based on time.

        Args:
            time (torch.Tensor): Time tensor.
            intensity (torch.Tensor): intensity tensor.

        Returns:
            torch.Tensor: Global intensity tensor.
        """
        intensity_delta = self.model(time.unsqueeze(0)
                       )
        intensity_delta = nn.functional.softplus(intensity_delta)

        return intensity_delta


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

    '''
    def forward(self, time, h_0 = 0):
        """
        Compute the global intensity based on time using ODE integration.

        Args:
            time (torch.Tensor): Time tensor.
            h_0 (int): Initial time. Default is 0.

        Returns:
            torch.Tensor: Global intensity tensor.
        """
        t = torch.tensor([h_0, time], dtype=torch.float32)  # Time points for the integration
        initial_cond = torch.tensor([0.0], requires_grad=True) # don't like that

        out = odeint(self.ode_func, initial_cond, t)[1]
        out = nn.functional.sigmoid(out)
        return out
    '''

    '''
    def forward(self, time, h_0 = 0, interval_time=torch.as_tensor([.2])):
        """
        Evaluate the integral of the intensity function over discrete time intervals.

        Args:
            time (torch.Tensor): Time tensor. (assume only 1 element)
            h_0 (int): Initial time. Default is 0.
            interval_time (torch.Tensor): Time step size. Default is 0.02.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        out = torch.zeros((1, 1), requires_grad=True)
        num_intervals = int(math.floor(time.item()/interval_time))

        for _ in range(num_intervals):
            out = out + (self.ode_func(interval_time, out) * interval_time).clone()
        
        rest_interval = time.item()%interval_time
        if MIN_INTEGRAL < rest_interval:# numerical issues
            out = out + (self.ode_func(rest_interval, out) * rest_interval).clone()
        #needs to be >= 0   maybe also <=1?
        #out = nn.functional.relu(out)
        #out = nn.functional.sigmoid(out)
        #out= torch.exp(-out)
        return out
    '''

    def forward(self, time):
        return self.ode_func(time)


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

        self.user_intensity_model = User_State_Intensity_Model(state_size, model_hyp["user_model_hyp"])
        self.global_model = global_Intensity_Model(time_size, model_hyp["global_model_hyp"])

    
    def forward(self, state, time_delta, start_time, state_model, interval_time= .02, user_params=None):
        """
        Compute the total intensity as the sum of user and global intensity models.

        Args:
            state (torch.Tensor): State tensor.
            time (torch.Tensor): Time tensor.
            state_model (Callable): Function for evolving the state.

        Returns:
            torch.Tensor: Total intensity tensor.
        """

        out_user = torch.tensor([[1e-25]],dtype=torch.float32, requires_grad=True)# prob, min value to avoid errors
        out_global = torch.zeros((1, 1), requires_grad=True)
        user_intensity = torch.zeros((1, 1), requires_grad=True)
        global_intensity = torch.zeros((1, 1), requires_grad=True)

        if user_params is None:
            state_model = state_model
        else:
            state_model = partial(state_model, user_params=user_params)

        num_intervals = int(math.floor(time_delta.item()/interval_time))
        curr_time = start_time
        for _ in range(num_intervals):
            user_intensity = user_intensity + (self.user_intensity_model(state, 
                interval_time, user_intensity) * interval_time).clone()
            global_intensity = global_intensity + (self.global_model(curr_time) * interval_time).clone()
            out_user = out_user + user_intensity
            out_global = out_global + global_intensity
            state = state_model(h=interval_time, state=state)# advance state
            curr_time = curr_time +interval_time

        #   extra step
        rest_interval = time_delta.item()%interval_time
        if MIN_INTEGRAL < rest_interval:
            user_intensity = user_intensity + (self.user_intensity_model(state, rest_interval, user_intensity) * rest_interval).clone()
            global_intensity = global_intensity + (self.global_model(curr_time) * rest_interval).clone()
            out_user = out_user + user_intensity
            out_global = out_global + global_intensity
            #state = state_model(h=interval_time, state=state)
        #intensity = torch.clamp(intensity, min=EPSILON, max=1-EPSILON)
        #intensity = smoothclamp(intensity, EPSILON, 1-EPSILON)
        #out = smoothclamp_0_1(intensity)
        out_global = 1- torch.exp(- out_global)
        out_user = 1- torch.exp(- out_user)
        overall = out_global + out_user
        #print(overall)
        #overall = smoothclamp_0_1(overall)
        overall = torch.clamp(overall, min=0, max=1)
        #print(overall)
        #print("clamped", intensity,"\tuser: ", user_intensity,"\tglobal: ", global_intensity,"\tsum: ", global_intensity+user_intensity)
        return overall, user_intensity, out_global


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

        #x = nn.functional.relu(x)
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
            recommendations (torch.Tensor): Batch of recommendation tensors. Batch size of 1 is assumed.

        Returns:
            torch.Tensor: Interaction outcomes tensor for each recommendation.
        """
        out = []
        recommendations = recommendations[0]
        for i in range(len(recommendations)):
            recommendation = recommendations[i].view(1,1)
            curr_out = self.single_interaction_model(state, recommendation)
            out.append(curr_out)
        out = torch.stack(out, dim=1)
        return out


#______________________________________jump_____________________________-
class Jump_Model_with_State_encode(Base_Model):
    """
    A model for computing jump sizes based on state and recommendation outcomes(get all outcomes).

    Args:
        state_size (int): Size of the state vector.
        num_outcomes (int): Number of outcome dimensions.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, num_outcomes: int, model_hyp: Dict):
        super(Jump_Model_with_State_encode, self).__init__(state_size + num_outcomes, state_size, model_hyp)
    
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


class Jump_Model_ratio(Base_Model):
    """
    A model for computing jump sizes based on state and recommendation outcomes(get all outcomes).

    Args:
        state_size (int): Size of the state vector.
        num_outcomes (int): Number of outcome dimensions.
        model_hyp (Dict): Dictionary containing model hyperparameters.
    """
    def __init__(self, state_size: int, model_hyp: Dict):
        super(Jump_Model_ratio, self).__init__(1, state_size, model_hyp)
    
    def forward(self, ratios):
        """
        Compute the jump size based on state and recommendation outcomes.

        Args:
            ratios (torch.Tensor): Ratio of positive outsomes to all outcomes [0,1].

        Returns:
            torch.Tensor: Jump size tensor.
        """

        x = self.model(ratios)# no activation needed?

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
                                            hyperparameter_dict["state_model"]["model_hyp"])

        self.intensity_model = Intensity_Model(self.state_size, time_size=self.time_size,
                                               model_hyp=hyperparameter_dict["intensity_model"]["model_hyp"])

        self.interaction_model = Interaction_Model(self.state_size, self.recommendation_size,
                                                   self.num_interaction_outcomes, hyperparameter_dict["interaction_model"]["model_hyp"])
        
        jump_model_interaction_size = self.num_interaction_outcomes * self.num_interaction_outcomes# could change if weuse summary stats
        self.jump_model = Jump_Model_with_State_encode(self.state_size, jump_model_interaction_size, hyperparameter_dict["jump_model"]["model_hyp"])

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
        overall_prob, user_prob, local_prob = self.intensity_model(self.state, time_delta, time, state_model=self.state_model)
        if return_all:
            return overall_prob, user_prob, local_prob
        return overall_prob

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
        self.state = self.state + self.jump_model(self.state, interactions)
    
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
        super(Conditioned_User_simmulation_Model, self).__init__(hyperparameter_dict)

        self.state_model = Conditioned_User_State_Model(self.state_size, hyperparameter_dict["user_params_size"],
                                            hyperparameter_dict["state_model"]["model_hyp"])

    
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
    
    def eval_intensity(self, time_delta, time, return_all: bool = False):
        """
        Evaluate the intensity function at a given time delta.

        Args:
            time_delta (torch.Tensor): Time delta tensor.
            time(torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Intensity tensor.
        """
        overall_intensity, user_intensity, local_intensity = self.intensity_model(self.state, 
                    time_delta, time, state_model=self.state_model, user_params=self.user_params)
        if return_all:
            return overall_intensity, user_intensity, local_intensity 
        return overall_intensity


#______________________________________complete models_____________________________

class all_in_one_model(nn.Module):
    """
    A neural network model that integrates time, state, and noise into a comprehensive framework
    for sequential predictions. The model can optionally encode time, and predict both the next 
    time step and the corresponding state.
    
    Args:
        model_hyp (Dict): A dictionary containing the model hyperparameters for the 
                          time, state, and (optionally) jump models.
        timecheat (bool, optional): If True, allows direct access to global time. Defaults to False.
        noise_size (int, optional): The size of the noise vector used for prediction. Defaults to 1.
    """
    def __init__(self, model_hyp: Dict, timecheat = False, noise_size:int = 11):
        super(all_in_one_model, self).__init__()
        self.state_size = model_hyp["state_size"]
        self.noise_size = noise_size
        self.timecheat=timecheat
        extra=0

        self.time_size= model_hyp.get("time_embedding_size",0)
        self.max_freq= model_hyp.get("max_freq",70)
        self.encode_time = self.time_size > 0
        if timecheat:
            extra =1
            if self.encode_time:
                extra += self.time_size

        self.time_model = Base_Model(self.state_size + noise_size + extra, 1, model_hyp["time_model"]["model_hyp"])

        if self.encode_time:
                self.embed = SignWaveEmbedding(self.time_size, max_freq=self.max_freq)
                self.state_model = Base_Model(self.state_size+self.time_size +1, 
                        self.state_size, model_hyp["state_model"]["model_hyp"])
        else:
            self.state_model = Base_Model(self.state_size+1, self.state_size, 
                    model_hyp["state_model"]["model_hyp"])

        if "jump_model" in model_hyp.keys():
            self.jump_model = Jump_Model_ratio(self.state_size, 
                model_hyp["jump_model"]["model_hyp"])

    def forward(self, state):#useless
        """
        Forward pass to predict the next time step and corresponding state.
        
        Args:
            state (torch.Tensor): The current state tensor.
        
        Returns:
            tuple: A tuple containing the predicted next time step and the corresponding state.
        """
        time_next_clapped = self.get_time(state)
        next_state = self.get_new_state(state, time_next_clapped)
        return time_next_clapped, next_state
    
    def get_time(self, state, global_time=None):
        """
        Predicts the next time step based on the current state, noise, and optional global time.
        
        Args:
            state (torch.Tensor): The current state tensor.
            global_time (torch.Tensor, optional): The global time tensor, required if timecheat is True.
        
        Returns:
            torch.Tensor: The predicted next time step.
        """
        noise = torch.rand((1, self.noise_size), requires_grad=True)
        #print(f"noise: {noise}")
        if self.timecheat:
            global_time = torch.tensor([[global_time]])
            if self.timecheat:
                time_emb=self.embed(global_time)
                state_and_noise = torch.cat((state, noise, time_emb, global_time), dim=1)
            else:
                state_and_noise = torch.cat((state, noise, global_time), dim=1)
        else:
            state_and_noise = torch.cat((state, noise), dim=1)

        time_next = self.time_model(state_and_noise)
        time_next = nn.functional.softplus(time_next)
        return time_next#smoothclamp(time_next) #torch.clamp(time_next, 0, 70)#smoothclamp(time_next)

    def get_new_state(self, state, time_next_clapped):
        """
        Predicts the next state based on the current state and the predicted next time step.
        
        Args:
            state (torch.Tensor): The current state tensor.
            time_next_clapped (torch.Tensor): The predicted next time step.
        
        Returns:
            torch.Tensor: The predicted next state.
        """
        if self.encode_time:
            time_emb=self.embed(time_next_clapped)
            time_and_state = torch.cat((state, time_emb, time_next_clapped), dim=1)
        else:
            time_and_state = torch.cat((state, time_next_clapped), dim=1)
        return self.state_model(time_and_state)
    
    def jump(self, state, rev_ratio):
        """
        Adjusts the state based on a jump model, useful for modeling sudden transitions.
        
        Args:
            state (torch.Tensor): The current state tensor.
            rev_ratio (torch.Tensor): A ratio representing the reverse jump effect.
        
        Returns:
            torch.Tensor: The adjusted state after applying the jump.
        """
        jump_value = self.jump_model(rev_ratio)
        return state + jump_value
    


class Toy_intensity_Generator(nn.Module):
    """
    A toy model for generating event intensities based on a user's state. The model evolves the state
    over time and computes the intensity of events, allowing for the generation of event times and paths.

    Args:
        hyperparameter_dict (Dict): A dictionary containing the model hyperparameters for the state 
                                    and intensity models.
    """
    def __init__(self, hyperparameter_dict: Dict):
        super(Toy_intensity_Generator, self).__init__()
        self.state_size = hyperparameter_dict["state_size"]
        self.user_state_model = User_State_Model(self.state_size, 
                    hyperparameter_dict["state_model"]["model_hyp"])
        self.intensity_model = User_State_Intensity_Model(self.state_size, 
            hyperparameter_dict["intensity_model"]["model_hyp"])
        
        #self.state = torch.zeros((1,self.state_size))

    def find_h(self, state, uniform_guess, interval_size=.1, max_iter=250):
        """
        Finds the optimal time interval `h` such that the cumulative intensity surpasses the target value.
        
        Args:
            state (torch.Tensor): The current state tensor.
            uniform_guess (torch.Tensor): A uniform random value used to set the target intensity.
            interval_size (float, optional): The size of each interval for state evolution. Defaults to 0.1.
            max_iter (int, optional): The maximum number of iterations. Defaults to 250.
        
        Returns:
            float: The computed time interval `h`.
        """
        intensity = torch.zeros((1,1), requires_grad=True)

        target = -torch.log(uniform_guess+EPSILON)
        curr_state = state
        for i in range(max_iter):
            curr_state = self.user_state_model(curr_state, interval_size, interval_time=0.05)
            intensity = intensity + (self.intensity_model(h = interval_size, 
                                intensity=intensity, state=curr_state)*interval_size).clone()
            
            if intensity > target:
                return interval_size * i
        return interval_size * max_iter

    def sample_one(self, state):
        """
        Samples a single event time by finding the optimal interval `h`.
        
        Args:
            state (torch.Tensor): The current state tensor.
        
        Returns:
            torch.Tensor: The sampled event time `h`.
        """

        u = torch.rand(1, requires_grad=False)# theoretically one should also do -u + 1
        
        h_optimized = torch.tensor([self.find_h(state, u)])
        return torch.flatten(h_optimized)
    
    def sample_path(self, num_samples = 10,state = None):
        """
        Samples a path of event times based on the evolving state.
        
        Args:
            num_samples (int, optional): The number of event times to sample. Defaults to 10.
            state (torch.Tensor, optional): The initial state tensor. Defaults to a zero tensor.
        
        Returns:
            torch.Tensor: A tensor of sampled event times.
        """
        if state is None:
            state = torch.zeros((1,self.state_size))
        path = []
        curr_time = 0
        with torch.no_grad():
            for _ in range(num_samples):
                h = self.sample_one(state=state)
                curr_time = curr_time + h
                path.append(curr_time)
                state = self.user_state_model(state=state, h=h)
        path = torch.stack(path)
        return path.detach().numpy()
    
    def evolve_state(self, state, delta):
        """
        Evolves the state by a given time interval `delta`.
        
        Args:
            state (torch.Tensor): The current state tensor.
            delta (float): The time interval by which to evolve the state.
        
        Returns:
            torch.Tensor: The evolved state tensor.
        """
        return self.user_state_model(state=state, h=delta)


class Toy_intensity_Comparer(nn.Module):
    """
    A neural network model designed to compare event intensities over time by evolving the user's state
    and calculating the resulting intensities. It supports different state evolution models(ODE or simple) and optional
    time encoding.

    Args:
        hyperparameter_dict (Dict): A dictionary containing the model hyperparameters for the state,
                                    intensity, and (optionally) jump models.
    """
    def __init__(self, hyperparameter_dict: Dict):
        super(Toy_intensity_Comparer, self).__init__()

        self.state_size = hyperparameter_dict["state_size"]
        self.time_size= hyperparameter_dict.get("time_embedding_size",0)
        self.max_freq= hyperparameter_dict.get("max_freq",70)
        self.encode_time = self.time_size > 0

        self.state_model_type = hyperparameter_dict["state_model_type"]
        if self.state_model_type == "simple":
            if self.encode_time:
                self.embed = SignWaveEmbedding(self.time_size, max_freq=self.max_freq)
                self.user_state_model = Base_Model(self.state_size+self.time_size+1, self.state_size,
                        hyperparameter_dict["state_model"]["model_hyp"],
                    )
            else:
                self.user_state_model = Base_Model(self.state_size+1, self.state_size,
                        hyperparameter_dict["state_model"]["model_hyp"],
                    )
        elif self.state_model_type == "ode":
            self.user_state_model = User_State_Model(self.state_size, 
                hyperparameter_dict["state_model"]["model_hyp"],)
        else:
            raise ValueError
        
        
        self.intensity_model = User_State_Intensity_Model_simple(self.state_size, 
            hyperparameter_dict["intensity_model"]["model_hyp"])
        
        if "jump_model" in hyperparameter_dict.keys():
            self.jump_model = Jump_Model_ratio(self.state_size, 
                hyperparameter_dict["jump_model"]["model_hyp"])

    def evolve_state(self, state, delta):
        """
        Evolves the user's state over a time interval `delta`.

        Args:
            state (torch.Tensor): The current state tensor.
            delta (torch.Tensor): The time interval over which to evolve the state.
        
        Returns:
            torch.Tensor: The evolved state tensor.
        """
        if self.state_model_type != "simple":
            return self.user_state_model(state, delta) 
        if self.encode_time:
            time_emb=self.embed(delta)
            input = torch.cat((state, time_emb, delta), dim=1)
        else:
            input = torch.cat((state, delta), dim=1)
        return self.user_state_model(input)
    
    def get_intensity(self, state):# might want to make this time dependent
        """
        Calculates the event intensity based on the current state.

        Args:
            state (torch.Tensor): The current state tensor.
        
        Returns:
            torch.Tensor: The calculated intensity.
        """
        return self.intensity_model(state) 
    
    def forward(self, state, times, return_new_state=False):
        """
        Computes the intensity after evolving the state over given time intervals.

        Args:
            state (torch.Tensor): The current state tensor.
            times (torch.Tensor): The time intervals over which to evolve the state.
            return_new_state (bool, optional): Whether to return the new state along with the intensity. Defaults to False.
        
        Returns:
            torch.Tensor: The calculated intensity.
            torch.Tensor (optional): The new state, if `return_new_state` is True.
        """
        new_state = self.evolve_state(state, times)
        
        freq =  self.get_intensity(new_state)
        if return_new_state: 
            return freq, new_state
        return freq
    
    def jump(self, state, rev_ratio):# not working well 
        """
        Adjusts the state based on a jump model, useful for modeling sudden transitions.

        Args:
            state (torch.Tensor): The current state tensor.
            rev_ratio (torch.Tensor): A ratio representing the reverse jump effect.
        
        Returns:
            torch.Tensor: The adjusted state after applying the jump.
        """
        jump_value = self.jump_model(rev_ratio)
        return state + jump_value


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
    start_time = 0.
    target_time = 1.
    new_state = state_model(state, target_time)
    print("new_state: ", new_state, "\t shape: ", new_state.shape)

    # intensity model
    intensity_state_dict = {"model_hyp": {"user_model_hyp": {"layer_width": [3,3]},
            "global_model_hyp": {"layer_width": [3,3]}}
        }
    time_tensor = torch.as_tensor([target_time])
    start_time_tensor = torch.as_tensor([start_time])
    intensity_model = Intensity_Model(state_size=state_size, time_size= time_size, model_hyp= intensity_state_dict["model_hyp"])
    overall_intensity, user_intensity, out_global = intensity_model(new_state, time_tensor, state_model=state_model,
                                start_time=start_time_tensor)
    print("intensity: ", overall_intensity, "\t shape: ", overall_intensity.shape)

    # interaction model
    interaction_state_dict = {"model_hyp": {"layer_width": [3,3]}
                    }

    recommendations = torch.randn((1,num_recomm, recom_dim))
    #recommendations = recommendations.unsqueeze(0)
    interaction_Model = Interaction_Model(state_size=state_size, recommendation_size=recom_dim,
                                        num_outcomes=num_interaction_outcomes, interaction_model_hyp=interaction_state_dict["model_hyp"])
    interactions = interaction_Model(new_state, recommendations)
    print("Interactions: ", interactions, "\t shape: ", interactions.shape)
    
    # backprop?
    true_interactions = torch.randn((1,num_recomm, num_interaction_outcomes))
    loss = nn.functional.mse_loss(interactions, true_interactions)
    loss.backward()

    # jump
    jump_state_dict = {"model_hyp": {"layer_width": [3,3]}
                        }

    jump_model = Jump_Model_with_State_encode(state_size, num_interaction_outcomes*num_recomm, jump_state_dict["model_hyp"])
    jump = jump_model(new_state, interactions)
    print("jump: ", jump, "\t shape: ", jump.shape)
    new_state = new_state + jump
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

import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax

class BaselineLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaselineLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1):
        super(BaselineMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(LinearModel, self).__init__()
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, output_dim)
        self.linear_activation = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        activations = self.linear_activation(x)
        x = self.linear_0(x)
        x = activations * x
        x = self.linear_1(x)
        return x


class ShallowModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, activation='relu', normalize=True):
        super(ShallowModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.activation_layers = nn.ModuleList()
        for _ in range(num_layers+1):
            self.activation_layers.append(nn.Linear(input_dim, hidden_dim))

            
        if activation == 'relu':
            self.activation_function = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            self.activation_function = nn.Identity()
        if normalize:
            self.normalizer = nn.Softmax(dim=1)
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        activations = [self.normalizer(self.activation_layers[i](x)) for i in range(len(self.activation_layers))]
        for i in range(len(self.layers)):
            x = activations[i] * self.layers[i](x)
        x = self.output_layer(x)
        return x
    
    
class DeepModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, activation='relu', normalize=True):
        super(DeepModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation_layers = nn.ModuleList()
        self.activation_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.activation_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if activation == 'relu':
            self.activation_function = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            self.activation_function = nn.Identity()
        if normalize:
            self.normalizer = nn.Softmax(dim=1)
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation_layers[i](x) * self.layers[i](x)
        x = self.output_layer(x)
        return x

class LinearModelWithGumbel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation_layer, gumbel):
        super(LinearModelWithGumbel, self).__init__()
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, output_dim)
        self.activation_layer = activation_layer
        self.gumbel = gumbel

    def forward(self, x):
        if self.gumbel: activations = gumbel_softmax(self.activation_layer(x)) 
        else: 
            activations = torch.round(self.activation_layer(x))
        x = self.linear_0(x)
        x = activations * x
        x = self.linear_1(x)
        return x
    
    
    
class LinearModelWithDropout(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
        super(LinearModelWithDropout, self).__init__()
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, output_dim)
        self.activation_layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_rate = dropout_rate
        # keep dropout in eval mode

    def forward(self, x):
        activations = nn.functional.dropout(self.activation_layer(x), p=self.dropout_rate)
        x = self.linear_0(x)
        x = activations * x
        x = self.linear_1(x)
        return x

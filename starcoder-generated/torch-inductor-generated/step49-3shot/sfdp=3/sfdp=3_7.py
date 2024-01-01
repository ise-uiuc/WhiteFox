
import torch
class Model(torch.nn.Module):
    def __init__(self, weight, bias, scale):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        weight = weight.reshape(32, 32)
        bias = bias.reshape(32)
        scale = scale.reshape(1)
        weight = weight * scale[0]
        self.linear.weight.data = weight
        self.linear.bias.data = bias
 
    def forward(self, x1):
        return self.linear(x1)

# Initializing the model
m = Model([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [20.0])

# Inputs to the model
x1 = torch.randn(1, 32)

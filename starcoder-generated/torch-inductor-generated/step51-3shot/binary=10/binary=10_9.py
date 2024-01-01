
class Model(torch.nn.Module):
    def __init__(self, weight_shape, bias_shape):
        super().__init__()
        self.linear = torch.nn.Linear(*weight_shape)
        self.addition_weight = torch.nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
  
    def forward(self, x1):
        return torch.nn.functional.linear(x1, self.linear.weight + self.addition_weight)

# Initializing the model
__weights_dict = dict()
weight_shape = (3, 4)
bias_shape = (4,)
__weights_dict['linear.weight'] = torch.randn(weight_shape)
__weights_dict['linear.bias'] = torch.randn(bias_shape)
m = Model(weight_shape, bias_shape)

# Setting the model weights
for name, param in m.named_parameters():
    if name in __weights_dict:
        param.data = __weights_dict[name]

# Inputs to the model
x1 = torch.randn(2, 3)
__output_dict = m.state_dict()
__output_dict['linear.weight'] = __weights_dict['linear.weight'] + 2

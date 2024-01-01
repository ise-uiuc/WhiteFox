
class Model(torch.nn.Module):
    def __init__(self, linear_weight: torch.Tensor, linear_bias: torch.Tensor, other: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(other.shape[1], 1)
        self.linear.weight.data = copy.deepcopy(linear_weight)
        self.linear.bias.data = copy.deepcopy(linear_bias)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
linear_weight = torch.randn(*m.linear.weight.data.shape)
linear_bias = torch.zeros(*m.linear.bias.data.shape)
other = torch.randn(1, *m.linear.weight.data.shape[1:])
m = Model(linear_weight, linear_bias, other)

# Inputs to the model
x2 = torch.randn(1, *m.linear.weight.shape[1:])


class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu6(v2)
        return v3
 
# Initializing the parameters
linear_weight = torch.randn(3, 8)
linear_bias = torch.randn(8)
other_weight = torch.rand(8)
other_bias = torch.rand(8)
 
# Initializing the model
m = Model(linear_weight, linear_bias)
m.register_buffer("other", other_weight + other_bias)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

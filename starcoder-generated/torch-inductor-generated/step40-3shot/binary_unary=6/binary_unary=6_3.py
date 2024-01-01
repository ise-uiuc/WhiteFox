
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = torch.nn.Linear(2,1)
        self.linear.weight = torch.nn.Parameter(weight)
        self.linear.bias = torch.nn.Parameter(bias)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        other = 1
        v2 = v1 - other
        v3 = F.relu(v2)
        return v3

# Initializing the model
weight = torch.randn(2,1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
m = Model(weight, bias)

# Inputs to the model
x1 = torch.randn(1, 2)

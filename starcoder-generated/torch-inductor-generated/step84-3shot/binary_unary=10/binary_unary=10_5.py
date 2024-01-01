
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)
        self.linear.weight = weight
        self.linear.bias = bias
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
weight = torch.nn.Parameter(torch.tensor([[1.0]]))
bias = torch.nn.Parameter(torch.tensor([0.0]))
m = Model(weight, bias)

# Inputs to the model
x1 = torch.randn(1, 1)

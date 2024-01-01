
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.full((2, 2), -0.05)
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.linear.weight = torch.nn.Parameter(weights, requires_grad=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 3.141593
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(2, 2)

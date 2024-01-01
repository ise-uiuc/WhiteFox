
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(128, 64)
        self.linear = torch.nn.Linear(64, 128, bias=False)
        self.linear.weight = torch.nn.parameter.Parameter(weight)
 
    def forward(self, x1, other: torch.Tensor):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
# A dummy data as the 'other' tensor
other = torch.randn(1, 128)

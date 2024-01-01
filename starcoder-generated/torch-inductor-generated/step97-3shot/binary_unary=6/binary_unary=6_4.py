
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.other = torch.nn.Parameter(torch.tensor(other), requires_grad=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(1, 8)

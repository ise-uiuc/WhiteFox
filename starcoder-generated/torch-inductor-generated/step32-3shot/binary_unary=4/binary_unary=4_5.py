
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.empty(1, 8, 64, 64).uniform_()

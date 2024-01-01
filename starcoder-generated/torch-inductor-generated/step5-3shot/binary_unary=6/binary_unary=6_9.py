
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Linear(64, 16)
 
    def forward(self, x):
        v1 = torch.relu(self.f(x))
        v2 = v1 - 0.2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(20, 64)

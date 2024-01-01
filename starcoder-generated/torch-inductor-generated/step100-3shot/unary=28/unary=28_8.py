
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
 
    def forward(self, x):
        v1 = self.fc1(x)
        return torch.clamp_min(torch.clamp_max(v1, min), max)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)

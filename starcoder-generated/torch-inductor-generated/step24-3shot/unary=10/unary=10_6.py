
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.fc2(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 64)

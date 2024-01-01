
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 100)
 
    def forward(self, x2):
        v2 = self.fc(x2)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(10000, 100)

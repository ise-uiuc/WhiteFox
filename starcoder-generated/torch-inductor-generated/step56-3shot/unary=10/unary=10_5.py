
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = torch.nn.Linear(16, 8)
 
    def forward(self, x0):
        v0 = self.fc_1(x0)
        v1 = v0 + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 16)

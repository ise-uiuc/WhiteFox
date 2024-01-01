
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(13, 5)
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = v1 + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 13)

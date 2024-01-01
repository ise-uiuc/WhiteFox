
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        return torch.clamp_max(torch.clamp_min(v2, 0), 6) / 6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)

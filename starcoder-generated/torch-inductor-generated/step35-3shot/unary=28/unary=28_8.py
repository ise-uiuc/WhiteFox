
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min=-5)
        v3 = torch.clamp_max(v2, max=5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 7)

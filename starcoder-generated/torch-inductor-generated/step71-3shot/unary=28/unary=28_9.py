
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, x2)
        v3 = torch.clamp_max(v2, x3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = 0.0
x3 = 2.0

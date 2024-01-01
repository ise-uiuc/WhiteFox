
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, 0.1)
        v3 = v2 > 0.9
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.lin = torch.nn.Linear(128, 64)
        self.min_value = min_value
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.flatten(1)
        v3 = self.lin(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, 0) # Clamp the output to 0
        return v5

# Initializing the model
m = Model(-8.0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 128, 4, 4)

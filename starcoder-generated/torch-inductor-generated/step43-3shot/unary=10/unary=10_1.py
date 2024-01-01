
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 3
        self.conv = torch.nn.Conv2d(num_channels, num_channels, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(num_channels, 4)
 
    def forward(self, x1):
        v1 = self.linear(self.conv(x1))
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

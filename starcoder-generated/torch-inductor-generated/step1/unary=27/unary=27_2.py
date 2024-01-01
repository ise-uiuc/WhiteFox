
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp(v1, -0.12918842429161072, 0.15983960728645325)
        v3 = torch.clamp(v2, 0.9573027329444885, float('inf'))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

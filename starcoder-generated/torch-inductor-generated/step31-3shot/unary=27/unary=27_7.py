
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 20, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 20, 4, stride=2, padding=1)
        
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.5
max = 1.3373875217437744
# Inputs to the model
x1 = torch.randn(1, 3, 500, 800)

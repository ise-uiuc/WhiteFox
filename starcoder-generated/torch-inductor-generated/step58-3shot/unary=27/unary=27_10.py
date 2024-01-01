
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d( 1,  1, 1, stride=1, padding=0)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = torch.clamp_max(v1, self.max)
        v2 = torch.clamp_min(v3, self.min)
        return v2
min = -1.37
max = 0.6
# Inputs to the model
x1 = torch.randn( 10,  1, 10, 10)


class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.padding = torch.nn.ZeroPad2d((0,1,0,0))
        self.conv = torch.nn.Conv2d(66, 33, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.padding(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.1
max = 0.2
# Inputs to the model
x1 = torch.randn(1, 66, 50, 50)

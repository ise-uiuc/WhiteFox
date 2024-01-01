
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min, out=torch.cuda.FloatTensor())
        v3 = torch.clamp_max(v2, self.max, out=torch.cuda.FloatTensor())
        return v3
min = 0.54
max = 0.2
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)


class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 2, stride=2, padding=0)
        self.min = torch.tensor([min])[0]
        self.max = torch.tensor([max])[0]
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.0
max = 0.5
# Inputs to the model
x1 = torch.randn(1, 64, 595, 799)

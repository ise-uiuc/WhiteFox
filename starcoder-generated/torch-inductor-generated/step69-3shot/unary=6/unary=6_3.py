
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bnc = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2)
    def forward(self, x1):
        v1 = self.bnc(self.conv(x1))
        v2 = self.conv(x1)
        v3 = torch.cat([v1,v2])
        v4 = self.relu(v3)
        v5 = torch.nn.functional.interpolate(v3, scale_factor=0.5,mode='nearest')
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

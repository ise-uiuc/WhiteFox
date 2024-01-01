
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.add = torch.nn.functional.pad(mode='constant', value='3')
        self.clamp_min = torch.nn.functional.pad(mode='constant', value='0')
        self.clamp_max = torch.nn.functional.pad(mode='constant', value='6')
        self.div = torch.nn.functional.pad(mode='constant', value='6')
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = self.add(v2)
        v4 = self.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

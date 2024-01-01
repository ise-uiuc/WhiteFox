
class Model(torch.nn.Module):
    def __init__(self, min, max, tanh_enabled):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, bias=True, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.min = min
        self.max = max
        self.tanh_enabled = tanh_enabled
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        if (self.tanh_enabled):
            v4 = self.tanh(v3)
        else:
            v4 = self.sigmoid(v3)
        return v4
min = 0
max = 2
tanh_enabled = False
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

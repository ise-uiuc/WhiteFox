
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.min = min
        self.max = max
    def forward(self, x2):
        v0 = x2
        v1 = self.conv1(v0)
        v2 = self.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = self.sigmoid(v3)
        v5 = v4 + 1.0 + 5.0
        v6 = torch.tanh(v5)
        v7 = torch.clamp_max(v6, self.max)
        v8 = torch.clamp_max(v7, 0.3)
        v9 = torch.clamp_max(v8, -0.3)
        v10 = self.sigmoid(v9)
        return v10
min = -5.0
max = -5.0
# Inputs to the model
x2 = torch.randn(1, 1, 1, 1)

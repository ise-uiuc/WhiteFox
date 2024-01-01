
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 16, 3, stride=1, padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(4, 32, 3, stride=1, padding=(1, 1))
        self.conv = torch.nn.Conv2d(32, 16, 3, stride=1, padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(16, 128, 3, stride=1, padding=(1, 1))
        self.linear = torch.nn.Linear(128, 256)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(x1)
        v5 = torch.stack((v3, v4), dim=1)
        v6 = self.conv(v5)
        v7 = torch.clamp_min(v6, min=self.min)
        v8 = torch.clamp_max(v6, max=self.max)
        v9 = torch.nn.functional.interpolate(v8, scale_factor=(2.0, 3.0), mode='linear', align_corners=True, recompute_scale_factor=False)
        v10 = self.conv4(v9)
        v11 = v10.flatten(1, -1)
        v12 = self.linear(v11)
        return v12
min = 0.3
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 7, 32, 56)

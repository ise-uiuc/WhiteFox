
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 128, 3, stride=(1, 2), padding=(0, 2))
        self.conv2 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 32, 3, stride=(1, 2), padding=(0, 1))
        self.conv4 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=(0, 1))
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        v2 = self.conv2(v1)
        v2 = torch.clamp_min(v2, self.min)
        v2 = torch.clamp_max(v2, self.max)
        v3 = self.conv3(v2)
        v3 = torch.clamp_min(v3, self.min)
        v3 = torch.clamp_max(v3, self.max)
        v4 = self.conv4(v3)
        v4 = torch.clamp_min(v4, self.min)
        v4 = torch.clamp_max(v4, self.max)
        return v4
min = 1e-10
max = -0.01
# Inputs to the model
x1 = torch.randn(1, 256, 32, 768)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, -0.9)
        v4 = self.conv3(v3)
        v5 = torch.clamp_min(v4, 0.8)
        v6 = self.conv4(v5)
        v7 = torch.clamp_min(v6, -0.3)
        v8 = self.conv5(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

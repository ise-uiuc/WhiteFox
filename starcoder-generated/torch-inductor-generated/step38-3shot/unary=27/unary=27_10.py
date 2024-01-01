
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=2, padding=4)
        self.conv2 = torch.nn.Conv2d(5, 8, 2, stride=5, padding=2)
        self.conv3 = torch.nn.ConvTranspose2d(8, 6, 9, stride=4, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv3(v6)
        return v7
min = 0.1
max = 0.5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(33, 8, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 33, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(33, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 33, 224, 224)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(5, 5, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, stride=2, padding=1)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.conv3 = torch.nn.Conv2d(5, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.act1(v1)
        v3 = self.conv2(v2)
        v4 = self.gap(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)

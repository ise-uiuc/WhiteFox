
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 23, 3, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(6, 46, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 64, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv2(v3)
        v5 = v3 + v4
        return v5 + v1
# Inputs to the model
x = torch.randn(1, 5, 8, 8)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, (1, 9), stride=1, padding=8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv2(v1)
        v5 = self.conv2(v2)
        v6 = self.conv2(v3)
        v7 = v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

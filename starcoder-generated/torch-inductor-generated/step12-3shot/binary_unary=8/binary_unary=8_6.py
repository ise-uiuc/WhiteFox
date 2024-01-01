
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 7, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(7, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 7, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = v1 + v2
        v5 = torch.relu(v4 + v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
x2 = torch.randn(1, 6, 64, 64)

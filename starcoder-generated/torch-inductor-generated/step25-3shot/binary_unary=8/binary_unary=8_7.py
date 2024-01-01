
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        t1 = v4 - 1
        v5 = torch.relu(t1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

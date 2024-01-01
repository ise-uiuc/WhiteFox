
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 32, 5, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 7, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 11
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = v5 - 13
        v7 = F.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1 - 0.1)
        v3 = F.max_pool2d(v2, 2)
        v4 = v3 - 0.5
        v5 = F.relu(v4)
        v6 = self.conv2(v5)
        v7 = torch.tanh(v6)
        v8 = v7 - 0.5
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)

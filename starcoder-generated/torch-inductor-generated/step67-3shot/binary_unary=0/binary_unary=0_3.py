
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 8, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(3, 3, 8, stride=1, padding=4)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = x3 + v1 * v2
        v5 = torch.relu(v4)
        v5 = torch.relu(v3)
        v6 = self.conv2(v5)
        v6 = 2.0 * v6
        v7 = v6 + v4
        v8 = torch.relu(v7)
        v9 = v8 * x3
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
x3 = torch.randn(1, 3, 224, 224)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.flatten1 = torch.nn.Flatten(1, 1)
        self.dense1 = torch.nn.Dense(8296, 10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = self.conv2(x1)
        v5 = self.flatten1(v1 + v2 + v3 + v4)
        v6 = self.dense1(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

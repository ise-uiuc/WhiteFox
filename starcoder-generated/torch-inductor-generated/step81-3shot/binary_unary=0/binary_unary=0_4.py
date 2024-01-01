
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, groups=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.sin(v2)
        v4 = v3 + x
        v5 = torch.relu(v4)
        v6 = v5 + v1
        v7 = torch.relu(v5)
        v8 = v6 + v7
        return v8
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

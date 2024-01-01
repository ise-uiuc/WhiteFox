
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = torch.concat([v1, v2], 1)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 100, 100)

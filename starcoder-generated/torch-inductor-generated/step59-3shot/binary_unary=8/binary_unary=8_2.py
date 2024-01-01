
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Parameter(torch.tensor(-10.0))
        self.v2 = torch.nn.Parameter(torch.tensor(-10.0))
        self.v3 = torch.nn.Parameter(torch.tensor(-10.0))
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.v1 + self.v2
        v2 = self.conv1(x1)
        v3 = self.conv2(v2)
        v4 = v1 + self.v3
        v5 = v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

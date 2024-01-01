
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(32, 4, 3)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.relu3(v1)
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 7, stride=2, padding=3, dilation=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 7, stride=2, padding=3, dilation=1)
        self.conv3 = torch.nn.Conv2d(2, 4, 7, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 13, 15)

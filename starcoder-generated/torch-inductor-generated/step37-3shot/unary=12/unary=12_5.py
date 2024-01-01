
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v2)
        v4 = self.flatten(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1, dilation=1)
        self.conv_1 = torch.nn.Conv2d(4, 6, 1, stride=1, padding=0, dilation=1)
        self.conv_2 = torch.nn.Conv2d(6, 2, 1, stride=1, padding=0, dilation=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_1(v1)
        v3 = self.conv_2(v2)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

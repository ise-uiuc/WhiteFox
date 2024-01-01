
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.relu_1 = torch.nn.ReLU(inplace=True)
        self.conv_2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.relu_2 = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        negative_slope = 0.3
        v1 = self.relu_1(self.conv_1(x))
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.relu_2(self.conv_2(v4))
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)

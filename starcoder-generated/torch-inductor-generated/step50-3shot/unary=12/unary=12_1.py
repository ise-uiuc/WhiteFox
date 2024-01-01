
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 12, 4, stride=4, padding=2, dilation=1)
        self.conv_2 = torch.nn.Conv2d(12, 64, 1, stride=1, padding=0)
        self.conv_3 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_2(v1)
        v5 = F.sigmoid(v4)
        v6 = self.conv_3(x1)
        v7 = (v6*v5)+(v3*v2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

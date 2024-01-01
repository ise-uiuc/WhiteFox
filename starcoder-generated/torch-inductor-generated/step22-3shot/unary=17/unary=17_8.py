
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(4, 12, 2, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(12, 8, [2, 4], stride=[1, 2], padding=[1, 3])
    def forward(self, x1):
        v1 = self.conv_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 7, 7)

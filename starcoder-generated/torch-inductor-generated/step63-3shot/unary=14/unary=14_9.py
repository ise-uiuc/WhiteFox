
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv3_4 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv3(x1)
        v2 = self.conv3_2(x2)
        v3 = v1 + v2
        v4 = torch.sigmoid(v3)
        v5 = self.conv3_4(v4)
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 128, 10, 10)
x2 = torch.randn(1, 128, 10, 10)
x3 = torch.randn(1, 128, 10, 10)

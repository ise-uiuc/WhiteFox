
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 4, (1, 1), stride=2, padding=0)
        self.conv_2 = torch.nn.Conv2d(4, 8, (1, 1), stride=2, padding=0)
        self.conv = torch.nn.Conv2d(8, 16, (1, 1), stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

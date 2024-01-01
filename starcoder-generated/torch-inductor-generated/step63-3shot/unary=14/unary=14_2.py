
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv8_38 = torch.nn.Conv2d(8, 8, 2, stride=1, padding=1)
        self.conv8_39 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv8_38(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv8_39(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)

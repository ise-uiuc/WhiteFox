
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3_1 = v1 * v2
        v4_1 = v2 * v1
        return v3_1, v4_1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

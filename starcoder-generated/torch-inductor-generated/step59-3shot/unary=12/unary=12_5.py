
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x2 = torch.randn(1, 16, 64, 64)

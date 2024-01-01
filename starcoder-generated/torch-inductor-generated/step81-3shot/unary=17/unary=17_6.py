
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(64)
        self.conv = torch.nn.ConvTranspose2d(64, 64, 1)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = torch.relu(v2)
        r = torch.sigmoid(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 5, 5)

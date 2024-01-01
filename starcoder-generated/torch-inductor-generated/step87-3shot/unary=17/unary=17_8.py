
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)
        self.conv = torch.nn.ConvTranspose2d(2, 2, 3, stride=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)

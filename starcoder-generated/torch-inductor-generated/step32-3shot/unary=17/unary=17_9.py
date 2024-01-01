
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 64, 1, stride=2)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

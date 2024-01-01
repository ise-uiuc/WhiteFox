
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.3899040662284746
        v3 = v1 * 0.5
        v4 = torch.erfinv(v3)
        v5 = torch.erf(v4)
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 35, 39)

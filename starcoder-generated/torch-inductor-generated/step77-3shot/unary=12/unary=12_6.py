
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.sig1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.sig2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sig1(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.sig2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)

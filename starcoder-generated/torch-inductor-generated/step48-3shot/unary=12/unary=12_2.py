
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
# Input1: Input tensor
# Input2: Output tensor

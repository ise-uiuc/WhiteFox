
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.multiply = torch.mul
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.multiply(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

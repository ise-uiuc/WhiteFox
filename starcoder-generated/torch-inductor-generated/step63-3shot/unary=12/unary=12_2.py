
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = torch.mul
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)

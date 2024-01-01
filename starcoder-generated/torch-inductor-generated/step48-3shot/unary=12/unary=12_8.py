
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1, stride=1, padding=0)
        self.pad = torch.nn.ZeroPad2d(2)
        self.sigmoid = torch.nn.Sigmoid()
        self.multiply = torch.mul
    def forward(self, x1):
        v1 = self.pad(x1)
        v2 = self.conv(v1)
        v3 = self.sigmoid(v2)
        v4 = v1 * v2
        return v3 * v4
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)

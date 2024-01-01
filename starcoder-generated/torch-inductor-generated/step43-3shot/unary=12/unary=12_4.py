
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.sigmoid = torch.sigmoid()
        self.mul = torch.mul(size=(1,1, 64, 64))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

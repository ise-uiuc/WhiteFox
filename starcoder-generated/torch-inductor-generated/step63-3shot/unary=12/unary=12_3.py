
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(256, 512, 3, stride=1, padding=1, dilation=2, groups=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = torch.mul
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 64)

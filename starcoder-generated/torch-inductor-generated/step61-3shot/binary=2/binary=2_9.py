
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, groups=1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.tanh(v1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)

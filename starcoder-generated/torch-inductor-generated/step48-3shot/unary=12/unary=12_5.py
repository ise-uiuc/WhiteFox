
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 4, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.max = torch.max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.tanh(v2)
        v4 = self.max(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)

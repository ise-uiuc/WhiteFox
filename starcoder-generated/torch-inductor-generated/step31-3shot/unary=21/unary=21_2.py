
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, v1):
        v2 = self.conv(v1)
        v3 = self.sigmoid(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
v1 = torch.randn(100, 64, 64, 64)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 1, stride=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.sigmoid(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 64, 64)

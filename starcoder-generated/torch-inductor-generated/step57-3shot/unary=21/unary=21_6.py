
class ModelTan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.hardtanh = torch.nn.Hardtanh()
        self.tanh_ = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.pool(x)
        v2 = self.conv(v1)
        v3 = self.hardtanh(v2)
        v4 = self.tanh_(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 256, 256)

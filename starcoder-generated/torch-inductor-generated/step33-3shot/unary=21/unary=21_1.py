
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 128, 1, stride=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x34):
        v1 = self.conv(x34)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x34 = torch.randn(1, 1, 28, 28)

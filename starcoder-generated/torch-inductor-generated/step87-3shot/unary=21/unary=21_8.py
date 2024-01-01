
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 3, stride=2, padding=(1, 1))
    def forward(self, x):
        a0 = torch.tanh(x)
        a1 = self.conv(a0)
        return a1
# Inputs to the model
X = torch.randn(1, 64, 56, 56)

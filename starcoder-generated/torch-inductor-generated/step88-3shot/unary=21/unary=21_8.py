
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=1, groups=8, padding=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x9):
        v1 = self.conv(x9)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x9 = torch.randn(3, 3, 256, 256)

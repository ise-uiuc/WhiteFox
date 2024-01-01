
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 12, 3)
    def forward(self, x6):
        x7 = self.conv(x6)
        x8 = torch.tanh(x7)
        return x8
# Inputs to the model
x6 = torch.randn(2, 12, 32, 32)

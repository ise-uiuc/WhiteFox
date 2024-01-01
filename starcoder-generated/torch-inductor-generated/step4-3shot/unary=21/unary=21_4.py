
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 64, 1, stride=1)
    def forward(self, x7):
        x8 = self.conv(x7)
        x9 = torch.tanh(x8)
        return x9
# Inputs to the model
x7 = torch.randn(1, 22, 128, 128)

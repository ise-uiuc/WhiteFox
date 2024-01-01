
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x28):
        v1 = self.conv(x28)
        v3 = v1 + 20.6958
        return torch.tanh(v1)
# Inputs to the model
x28 = torch.randn(1, 1, 32, 224)

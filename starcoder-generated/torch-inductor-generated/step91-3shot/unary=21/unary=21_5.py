
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 5, stride=1, padding=10)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x3 = torch.randn(1, 3, 32, 32)

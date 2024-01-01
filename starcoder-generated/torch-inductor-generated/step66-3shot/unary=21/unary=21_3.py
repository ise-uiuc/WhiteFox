
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x2):
        v2 = self.conv(x2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)

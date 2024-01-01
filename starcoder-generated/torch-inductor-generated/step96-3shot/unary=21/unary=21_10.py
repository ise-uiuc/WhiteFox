
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(torch.tanh(torch.tanh(v1)))
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

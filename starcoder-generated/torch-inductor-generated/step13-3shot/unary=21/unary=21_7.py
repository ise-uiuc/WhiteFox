
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 12, 3)
    def forward(self, x10):
        v11 = self.conv(x10)
        v12 = torch.tanh(v11)
        return v12
# Inputs to the model
x10 = torch.randn(3, 14, 512, 512)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.tanh(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

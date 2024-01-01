
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 4, 4)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
tensor = torch.randn(1, 3, 16, 16)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3000, 32, stride=1)
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(3000, 3000, 4, stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.conv1(x)
        x = self.tanh(x)
    return x
# Inputs to the model
x = torch.randn(1, 1, 64, 32)

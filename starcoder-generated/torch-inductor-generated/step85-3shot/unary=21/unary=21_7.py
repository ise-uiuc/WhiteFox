
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

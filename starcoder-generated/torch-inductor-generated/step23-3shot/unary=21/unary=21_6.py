
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, [3, 3], padding=[0, 0])
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 1, 1)

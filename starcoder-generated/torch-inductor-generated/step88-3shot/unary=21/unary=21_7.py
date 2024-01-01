
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=2, stride=8, padding=3, dilation=12)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 50, 50)

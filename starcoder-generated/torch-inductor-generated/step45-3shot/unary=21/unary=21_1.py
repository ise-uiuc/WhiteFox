
class PatternModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, padding=1, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

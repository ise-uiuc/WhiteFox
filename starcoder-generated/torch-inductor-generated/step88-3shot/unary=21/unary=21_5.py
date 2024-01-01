
class MyModuleTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)

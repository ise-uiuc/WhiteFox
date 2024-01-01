
class PatternModule(torch.nn.Module):
    def __init__(self):
        super(PatternModule, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=16, out_channels=16,
                                     kernel_size=1, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        y1 = self.conv(x1)
        z1 = self.tanh(y1)
        return z1
# Inputs to the model
x1 = torch.randn(2, 16, 32, 32)

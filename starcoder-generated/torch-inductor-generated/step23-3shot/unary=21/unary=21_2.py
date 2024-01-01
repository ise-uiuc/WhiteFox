
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 256, 1, stride=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x2):
        x2 = self.conv(x2)
        v1 = self.tanh(x2)
        return v1
# Inputs to the model
x2 = torch.randn(1024, 64, 8, 8)

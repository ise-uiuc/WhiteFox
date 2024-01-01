
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 2, dilation=2, stride=2, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x8):
        v1 = self.conv(x8)
        t1 = self.tanh(v1)
        return t1
# Inputs to the model
x8 = torch.randn(1, 3, 256, 256)

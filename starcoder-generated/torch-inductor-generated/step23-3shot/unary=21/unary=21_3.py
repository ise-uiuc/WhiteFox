
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=(2, 2), padding=9, dilation=1, groups=1, bias=None)
        self.tanh = torch.nn.Tanh()
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x0 = torch.randn(255, 3, 256, 256)

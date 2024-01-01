
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 3), stride=(1, 2), dilation=(1, 3), padding=(1, 3), groups=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(x2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 86, 26)

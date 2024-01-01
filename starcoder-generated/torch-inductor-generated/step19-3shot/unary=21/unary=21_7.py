
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, stride=1, bias=None)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(255, 3, 256, 256)

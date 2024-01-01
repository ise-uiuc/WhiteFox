
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 256, 2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        t1 = self.conv(x1)
        y1 = self.tanh(t1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)

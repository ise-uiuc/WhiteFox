
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, strides=2, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1, other=1.0, padding1=True):
        v1 = self.conv(x1)
        v2 = self.tanh(v1) + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

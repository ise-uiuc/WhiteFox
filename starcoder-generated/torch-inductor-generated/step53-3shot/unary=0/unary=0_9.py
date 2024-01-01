
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=6)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = torch.tanh(v1)
        return v1
# Inputs to the model
x4 = torch.randn(4, 16, 1, 1)

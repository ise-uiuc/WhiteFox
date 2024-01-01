
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 1, 1, stride=1)
    def forward(self, x1, padding1=None, padding2=None):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)

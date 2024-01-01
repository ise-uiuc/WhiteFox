
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(66, 2, 1, stride=1, padding=0)
    def forward(self, x1, other=torch.randn(1, 2, 18, 18), padding1=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 66, 64, 64)

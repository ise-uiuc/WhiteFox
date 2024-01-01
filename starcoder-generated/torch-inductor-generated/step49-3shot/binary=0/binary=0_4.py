
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None):
        v1 = self.conv1(x1)
        if not padding1 is None:
            v1 += padding1
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)

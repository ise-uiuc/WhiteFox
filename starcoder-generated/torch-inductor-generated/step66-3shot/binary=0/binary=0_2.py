
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, other=1, padding2=True):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 16, 16)

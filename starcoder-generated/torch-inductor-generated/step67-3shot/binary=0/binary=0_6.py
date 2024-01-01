
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 7, stride=2, padding=1)
    def forward(self, x1, other=False):
        v1 = self.conv(x1)
        if other:
            other = torch.randn(v1.shape)
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

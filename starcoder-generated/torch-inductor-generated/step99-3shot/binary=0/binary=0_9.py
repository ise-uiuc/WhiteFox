
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 5, 3, stride=1, padding=0)
    def forward(self, x1, other=5):
        v1 = self.conv(x1)
        if other == 5:
            other = torch.randn(v1.shape)
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(3, 7, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, groups=1)
    def forward(self, x):
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)

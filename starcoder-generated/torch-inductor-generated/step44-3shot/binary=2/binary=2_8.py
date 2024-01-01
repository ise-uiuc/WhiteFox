
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, 1, 0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.62
        return v2
# Inputs to the model
x = torch.randn(1, 1, 5, 1)

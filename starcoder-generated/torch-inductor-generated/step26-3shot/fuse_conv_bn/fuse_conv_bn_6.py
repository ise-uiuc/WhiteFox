
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        y = v1 + v1
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)

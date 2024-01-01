
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        y = self.conv(x)
        return x + y
# Inputs to the model
x = torch.randn(1, 3, 6, 6)

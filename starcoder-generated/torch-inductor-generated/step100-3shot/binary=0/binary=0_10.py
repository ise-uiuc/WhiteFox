
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (4, 6), stride=1, padding=3)
    def forward(self, x1, padding=True):
        x2 = self.conv(x1)
        x3 = x2 + x2
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 124, 192)

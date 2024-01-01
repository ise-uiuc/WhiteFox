
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.conv = torch.nn.Conv2d(3, 3, 3, groups=3)
    def forward(self, x1):
        s = self.conv(x1)
        return s + s
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)

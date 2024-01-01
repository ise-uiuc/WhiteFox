
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 2, stride=2, padding=2)
    def forward(self, x1):
        x2 = self.conv(x1 + torch.randn(3, 3, 28, 28) / 3)
        return x2 + 3
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)

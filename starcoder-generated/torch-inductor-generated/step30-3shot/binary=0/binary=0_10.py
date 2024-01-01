
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.weight = torch.randn(self.conv.weight.shape)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 + self.weight
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

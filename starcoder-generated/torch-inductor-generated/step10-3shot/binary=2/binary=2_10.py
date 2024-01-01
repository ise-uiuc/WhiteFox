
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        d1 = self.conv(x1)
        d2 = d1 - 0.102
        return torch.cat((d1, d2), 0)
# Inputs to the model
x1 = torch.randn(1, 1, 192, 182)

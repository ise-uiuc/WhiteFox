
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (1, 1), stride=1, padding=1)
    def forward(self, x, other=1):
        v1 = self.conv(x)
        v2 = v1 + other
        return v2
# Inputs to the model
x = torch.randn(1, 1, 28, 28)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 3, padding=(1, 1), groups=4)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 5
        return v2
# Inputs to the model
x = torch.randn(1, 4, 32, 32)

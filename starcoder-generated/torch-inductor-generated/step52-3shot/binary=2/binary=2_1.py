
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - 19
        return v2
# Inputs to the model
x3 = torch.randn(2, 1, 987, 123)

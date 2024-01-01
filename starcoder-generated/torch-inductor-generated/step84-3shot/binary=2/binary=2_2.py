
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 512, (7, 7), (1, 1), (0, 0), 1, 1, False, False, False)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - -0.004375783195018279
        return v2
# Inputs to the model
x3 = torch.randn(1, 1, 1024, 1024)

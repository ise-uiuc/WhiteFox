
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3, stride=1, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v3 = 1.6 * v1 - torch.randn(2, 2, 3, 3) * v1
        return v3
# Inputs to the model
x3 = torch.randn(1, 3, 96, 96)

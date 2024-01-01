
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - torch.tensor(6.2)
        return v2
# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)

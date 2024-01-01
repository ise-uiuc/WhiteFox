
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, padding=1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1, 3)
        v2 = self.conv(v1)
        v3 = v2.permute(0, 1, 3, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

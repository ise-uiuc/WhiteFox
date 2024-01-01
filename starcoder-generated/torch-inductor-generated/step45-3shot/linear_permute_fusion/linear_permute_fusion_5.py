
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (2, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.permute(0, 2, 3, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)

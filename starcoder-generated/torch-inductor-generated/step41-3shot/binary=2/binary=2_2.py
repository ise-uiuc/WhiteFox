
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.full_like(v1, -1029)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

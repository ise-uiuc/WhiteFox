
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, 2, 1, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.mean(v1, dim=(2, 3))
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8, device='cpu')

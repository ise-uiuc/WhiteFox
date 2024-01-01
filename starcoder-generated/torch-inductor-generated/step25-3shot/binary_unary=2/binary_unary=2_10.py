
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1, groups=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        assert torch.numel(v1) == 1440
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

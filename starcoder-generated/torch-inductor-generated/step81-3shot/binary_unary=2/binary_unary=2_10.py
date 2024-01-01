
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 3, 7, stride=1, padding=0, groups=2, bias=False)
    def forward(self, x1):
        v2 = self.conv(x1 + x1)
        v1 = v2 - 1.5
        v3 = F.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 18, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 512, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.0
        v3 = v2 * 2.0
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 256, 28, 28)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 1, 1, stride=1, padding=0)
    def forward(self, x1, other=19):
        v1 = self.conv(x1)
        v3 = v1.add(other, alpha=1)
        v2 = v3.add(other)
        return v2
# Inputs to the model
x1 = torch.randn(1, 14, 224, 224)

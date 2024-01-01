
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        v1 = self.bn(x)
        v2 = F.interpolate(v1, size=(112, 96))
        v3 = v2.permute(0, 2, 3, 1)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

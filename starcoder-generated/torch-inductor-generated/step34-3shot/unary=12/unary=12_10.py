
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, groups=3)
        self.sig = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sig(v1)
        v3 = torch.mean(v1, dim=[1])
        v4 = torch.softmax(v3, dim=1)
        v5 = v2 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

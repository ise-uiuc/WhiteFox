
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 1, stride=2, padding=0, dilation=1)
    def forward(self, x1, other, padding3=None, padding4=None, padding5=None):
        v1 = self.conv(x1)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        v2 = torch.flatten(v1)
        v3 = v2.relu()
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 8, 8)

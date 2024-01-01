
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 14, 3)
    def forward(self, x):
        v1 = x.view(int(x.size(0)*2), int(x.size(1)/2), 6, 6)
        v2 = x.view(int(x.size(0)/2), int(x.size(1)*2), 6, 6)
        v3 = self.conv(v1)
        v4 = v2.mul(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 9, 4, 4)

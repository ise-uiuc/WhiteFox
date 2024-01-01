
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 4, stride=2, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(2, 2, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = v4.sigmoid()
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)

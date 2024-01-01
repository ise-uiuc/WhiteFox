
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(54, 30, 11, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 54, 61, 66)

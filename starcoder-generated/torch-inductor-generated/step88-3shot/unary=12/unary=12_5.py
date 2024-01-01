
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        return v3.transpose(2, 4).view([-1, 12, 1000])
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.add(v1, 0.14)
        v2 = F.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)

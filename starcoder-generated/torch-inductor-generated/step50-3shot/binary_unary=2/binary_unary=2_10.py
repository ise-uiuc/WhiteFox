
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 50, 5, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.1
        out = F.sigmoid(v2)
        return out
# Inputs to the model
x1 = torch.randn(1, 20, 28, 28)

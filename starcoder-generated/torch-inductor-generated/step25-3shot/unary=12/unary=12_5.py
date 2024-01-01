
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 3, stride=1, padding=0, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 64, 144, 160)

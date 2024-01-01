
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, 8, padding=8)
    def forward(self, x1):
        ret = F.sigmoid(x1)
        ret = self.conv(ret)
        return ret
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)

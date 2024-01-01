
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 1, stride=1, padding=1)
    def forward(self, x, padding1=0.5):
        v1 = self.conv(x)
        res = v1 + padding1
        return res
# Inputs to the model
x = torch.randn(1, 5, 64, 64)

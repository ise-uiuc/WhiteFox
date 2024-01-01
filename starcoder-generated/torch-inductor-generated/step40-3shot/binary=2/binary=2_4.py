
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1).reshape(256)
        t2 = t1 - 0.6264
        v1 = t2.view(1, 1, 16, 16)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

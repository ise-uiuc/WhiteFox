
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        t1 = torch.randn(v1.shape)
        t2 = torch.randn(v2.shape)
        v3 = v1 + t1
        v4 = v2 + t2
        return v3, v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 128, 128)

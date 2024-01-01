
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 9, 1, stride=1, padding=1)
    def forward(self, x1, y=False):
        v1 = self.conv(x1)
        if y:
            y = torch.randn(v1.shape)
            z = y.clone()
        else:
            z = torch.randn(2, 3)
        v2 = v1 + z
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)

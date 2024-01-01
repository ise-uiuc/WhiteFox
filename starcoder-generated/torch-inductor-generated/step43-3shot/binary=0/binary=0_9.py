
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 1, 1, stride=1, padding=1)
    def forward(self, x1, other1=None, other2=None):
        v1 = self.conv(x1)
        if other1 is None and other2 is None:
            other1 = torch.randn(v1.shape)
            other2 = torch.randn(v1.shape)
        v2 = v1 + other1
        v3 = v2 + other2
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)

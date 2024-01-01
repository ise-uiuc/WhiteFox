
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 1, stride=1, padding=1)
    def forward(self, x1, other1=1, other2=2):
        v1 = self.conv(x1)
        v2 = v1 + other1
        v3 = v2
        if other2 is None:
            v4 = v3
        else:
            v4 = v3 + other2
            v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)

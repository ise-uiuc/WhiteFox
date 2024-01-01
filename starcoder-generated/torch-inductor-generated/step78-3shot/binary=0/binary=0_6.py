
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        if other == 1:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2.add(1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 14, 14)

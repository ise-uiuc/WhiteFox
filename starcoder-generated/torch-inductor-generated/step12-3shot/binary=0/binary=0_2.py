
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 10, 1, stride=1, padding=1)
    def forward(self, x1, other=5):
        v1 = self.conv(x1)
        if other == 5:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(3, 17, 64, 64)

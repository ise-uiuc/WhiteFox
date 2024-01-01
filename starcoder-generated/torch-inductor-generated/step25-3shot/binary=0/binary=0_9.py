
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 3, 1, stride=1, padding=1)
    def forward(self, x1, bias1=6, other=None):
        v1 = self.conv(x1)
        if bias1 == 6 and other is None:
            other = torch.randn(v1.shape)
            bias1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)

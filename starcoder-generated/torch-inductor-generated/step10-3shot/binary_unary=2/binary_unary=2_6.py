
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=2, dilation=1, padding=3, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

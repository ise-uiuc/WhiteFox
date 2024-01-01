
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, other = 1):
        v1 = self.conv(x1)
        if other == 1:
            other = self.conv_2(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)

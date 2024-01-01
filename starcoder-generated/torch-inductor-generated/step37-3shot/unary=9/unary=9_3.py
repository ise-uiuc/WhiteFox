
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = 3 + self.conv_1(x1)
        v2 = v1.clamp_(0, 5)
        v3 = v2.div(6)
        v4 = v3.reshape(1, 8, 64, 64)
        v5 = self.conv_2(v4)
        v6 = v5.abs()
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

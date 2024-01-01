
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_b = torch.nn.Conv2d(3, 4, 1, stride=3, padding=0, dilation=2)
        self.conv_a = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_b(x1)
        v2 = self.conv_a(v1)
        v3 = torch.sigmoid(v2)
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)

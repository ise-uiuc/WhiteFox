
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv_3 = torch.nn.Conv2d(1, 1, 1, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv_2(x1)
        v2 = self.conv_3(x1)
        v3 = v2 * v1
        v4 = v1 * v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 125, 67)

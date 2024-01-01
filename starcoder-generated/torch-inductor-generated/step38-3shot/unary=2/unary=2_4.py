
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(18, 20, 9, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(20, 16, 7, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 18, 3, 9)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = self.conv_2(v1)
        return v1 - v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

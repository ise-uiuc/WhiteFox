
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0, dilation=2)
    def forward(self, input):
        v = self.conv(input)
        v2 = v.conv
        v3 = v2 - 2
        return v3
# Inputs to the model
input = torch.randn(2, 3, 64, 64)

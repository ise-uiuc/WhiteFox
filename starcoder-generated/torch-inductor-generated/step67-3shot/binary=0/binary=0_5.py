
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
    def forward(self, in1):
        x1 = self.conv(in1)
        r1 = x1 + x1
        return r1
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)

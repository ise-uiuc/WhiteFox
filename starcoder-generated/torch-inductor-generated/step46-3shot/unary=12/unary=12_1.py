
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1)
        self.flatten = torch.flatten
        self.linear = torch.nn.Linear((64*3*3), 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.flatten(v1)
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

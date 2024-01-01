
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=0)
    def forward(self, x1, v2, other=None):
        v1 = self.conv(x1)
        v2 = v2 + self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
v2 = torch.randn(1, 7, 20, 20)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x0, other=0, x1=None):
        if x1 == None:
            x1 = self.conv(x0)
        v1 = x1 + other
        return v1
# Inputs to the model
x0 = torch.randn(3, 3, 64, 64)

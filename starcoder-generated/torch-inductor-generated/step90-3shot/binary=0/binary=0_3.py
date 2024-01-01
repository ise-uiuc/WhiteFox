
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 7, 1, stride=1, padding=1)
    def forward(self, x1, x2, other1=None, other2=None):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        if other1 == None:
            other1 = torch.randn(v1.shape)
        v3 = v1 + other1
        if other2 == None:
            other2 = torch.randn(v2.shape)
        v4 = v2 + other2
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(1, 5, 30, 30)
x2 = torch.randn(1, 5, 30, 30)

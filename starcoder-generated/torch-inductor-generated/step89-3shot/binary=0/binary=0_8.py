
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7)
    def forward(self, x1, x2=None, y=2, z=7):
        y += 1
        v1 = self.conv1(x1)
        y -= 1
        if x2 == None:
            x2 = torch.randn(v1.shape)
        y += 1
        y = y // 2
        v2 = v1 + x2
        y *= 2
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
other = 1

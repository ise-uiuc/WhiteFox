
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7)
    def forward(self, x1, x2=0):
        v1 = self.conv(x1)
        if x2 == 0:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)

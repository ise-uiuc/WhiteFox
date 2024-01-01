
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 9, 7)
    def forward(self, x1, other=None):
        if other == None:
            other = torch.randn(x1.shape[-3], x1.shape[-2], 1)
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 17, 17)

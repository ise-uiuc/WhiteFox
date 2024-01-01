
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
    def forward(self, x1, other=False):
        v1 = self.conv(x1)
        if other == False:
            other = torch.randn(v1.shape)
        else:
            return v1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)

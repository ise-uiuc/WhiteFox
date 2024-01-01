
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 25, 1, stride=1, padding=0)
    def forward(self, x1, other=1, strides1=2):
        v1 = self.conv(x1)
        if strides1 == 2 and other == 1:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(2, 10, 64, 64)

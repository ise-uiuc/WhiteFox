
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 - other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 128, 128) # Please make sure that there is no padding in the convolution

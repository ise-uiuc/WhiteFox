
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=2, bias=False)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v2 = v1
        if other == True:
            other = torch.randn(v2.shape)
        if other.ndim >= 3 and other.shape[1] == x1.shape[1]:
            v2 = v2 + other
        elif other.ndim == 4 and other.shape[0] == x1.shape[1] and (other.shape[2] == v2.shape[2] and other.shape[3] == v2.shape[3]):
            v2 = v2 + other
        else:
            v2 = v1 + torch.randn(v1.shape)
        v2 = torch.flatten(v2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

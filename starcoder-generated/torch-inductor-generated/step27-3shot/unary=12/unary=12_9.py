
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(640, 160, 16, groups=4)  # output tensor is of size (N, 4, 1, 1)
        self.sigmoid = torch.nn.Sigmoid()  # output tensor is of size (N, 4, 1, 1)
        self.mul = torch.nn.Mul()  # output tensor is of size (N, 160, 1, 1). The input tensor sizes of Mul operation are (N, 4, 1, 1) and (N, 4, 1, 1)
    def forward(self, x1):
        v = self.conv(x1)
        v1 = self.sigmoid(v)
        v2 = self.mul(v, v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 640, 64, 64)

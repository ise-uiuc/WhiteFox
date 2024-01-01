
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 1, stride=1, padding=1)
    def forward(self, x1, other=1.0):
        v1 = self.conv(x1)
        if x1.is_cuda:
            other = other.cuda()
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1)

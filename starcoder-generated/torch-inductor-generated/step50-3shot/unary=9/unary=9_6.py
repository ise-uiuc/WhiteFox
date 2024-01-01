
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.randn(8, 3, 3, 3)
        bias = torch.randn(8)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=True)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=(1,1), bias=False)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=(1,1,1), bias=False)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=(1,1,0,1), bias=False)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=(0,1,1,1), bias=False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
    def forward(self, x1):
        t0 = self.conv(x1)
        t1 = t0.add_(3)
        t2 = t1.clamp_(-6, 6)
        t3 = t2.div_(6)
        return t3
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sigm = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = self.sigm(v1)
        return v2

# Inputs to the mode
x1 = torch.randn(1, 64, 32, 32)

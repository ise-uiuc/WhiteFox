
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.avg_pool_1 = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=1)
        self.avg_pool_2 = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = self.avg_pool_1(v1)
        v3 = self.avg_pool_2(v1)
        v4 = torch.tanh(v2)
        v5 = torch.mul(v2, v3)
        v6 = torch.add(v4, v5)
        v7 = torch.mul(x1, v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 18, 18)

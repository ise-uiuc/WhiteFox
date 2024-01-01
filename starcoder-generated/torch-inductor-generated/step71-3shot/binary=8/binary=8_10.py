
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg1 = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.avg2 = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.avg1(x1)
        v2 = self.avg2(x2)
        v3 = v1 + v2
        v4 = self.avg1(v3)
        v5 = self.avg2(v3)
        v6 = v4 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 72, 72)
x2 = torch.randn(1, 3, 72, 72)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=1)
        self.avg1 = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.avg(x)
        v2 = self.avg1(v1)
        if v2.ndim == 3:
            v2 = v2 - -7.50
        else:
            v2 = v2 - -7.50
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

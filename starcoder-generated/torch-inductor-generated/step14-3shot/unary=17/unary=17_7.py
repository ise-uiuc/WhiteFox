
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Sequential(Module0(), Module1())
    def forward(self, x1):
        v1 = self.module(x1)
        return v1


class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.max_pool2d(x1, [2, 3])
        return v1

class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.adaptive_avg_pool2d(x1, [2, 7])
        return v1

# Inputs to the model
x1 = torch.randn(1, 8, 96, 9)

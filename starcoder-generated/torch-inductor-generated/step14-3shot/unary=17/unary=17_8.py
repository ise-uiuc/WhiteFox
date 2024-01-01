
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.Sigmoid(), torch.nn.Conv2d(3, 10, [3, 3], stride=[1, 2], groups=32), torch.nn.Sigmoid())
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 97, 95)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(16, 16, [3, 2], stride=[1, 2], padding=(1, 1)), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 96, 9)

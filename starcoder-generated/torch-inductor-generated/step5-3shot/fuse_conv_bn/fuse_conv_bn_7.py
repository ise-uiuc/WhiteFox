
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(0)
        c.weight = torch.nn.Parameter(torch.randn(c.weight.shape))
        torch.manual_seed(1)
        c.bias = torch.nn.Parameter(torch.randn(c.bias.shape))
        bn = torch.nn.BatchNorm2d(3)
        bn.running_mean = torch.arange(3, dtype=torch.float)
        bn.running_var = torch.arange(3, dtype=torch.float) * 2 + 1
        self.layer = torch.nn.Sequential(c, bn)
    def forward(self, x):
        v = self.layer(x)
        return v
# Inputs to the model
x = torch.randn(2, 3, 4, 4)

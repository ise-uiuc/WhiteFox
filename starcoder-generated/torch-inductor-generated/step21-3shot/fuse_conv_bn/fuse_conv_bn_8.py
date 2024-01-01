
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(2, 4, 3)
        torch.manual_seed(3)
        c.weight = torch.nn.Parameter(torch.randn(c.weight.shape))
        torch.manual_seed(4)
        c.bias = torch.nn.Parameter(torch.randn(c.bias.shape))
        bn = torch.nn.BatchNorm2d(4)
        bn.running_mean = torch.arange(4, dtype=torch.float)
        bn.running_var = torch.arange(4, dtype=torch.float) * 2 + 1
        bn.affine = False
        self.layer = torch.nn.Sequential(c, bn)
    def forward(self, x1):
        v1 = self.layer(x1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 2, 4, 4)

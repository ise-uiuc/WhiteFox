
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        a = torch.nn.modules.ModuleList([torch.nn.Conv2d(2, 2, 3) for _ in range(3)])
        torch.manual_seed(3)
        a[1].weight = torch.nn.Parameter(torch.randn(a[1].weight.shape))
        torch.manual_seed(4)
        a[2].weight = torch.nn.Parameter(torch.randn(a[2].weight.shape))
        torch.manual_seed(5)
        a[1].bias = torch.nn.Parameter(torch.randn(a[1].bias.shape))
        torch.manual_seed(6)
        a[2].bias = torch.nn.Parameter(torch.randn(a[2].bias.shape))
        self.layer = a
    def forward(self, x1):
        s1 = self.layer[1](x1)
        t1 = self.layer[2](x1)
        return s1+t1
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)

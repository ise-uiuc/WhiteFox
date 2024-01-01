
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(2, 4, 4)
        torch.manual_seed(3)
        c.weight = torch.nn.Parameter(torch.randn(c.weight.shape))
        torch.manual_seed(4)
        c.bias = torch.nn.Parameter(torch.randn(c.bias.shape))
        self.c = c
    def forward(self, x):
        v = self.c(x)
        return v
# Inputs to the model
x1 = torch.randn(2, 2, 4, 4)

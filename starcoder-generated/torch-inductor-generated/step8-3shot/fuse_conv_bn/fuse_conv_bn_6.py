
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.BatchNorm2d(4))
        torch.manual_seed(3)
        s[0].weight = torch.nn.Parameter(torch.randn(s[0].weight.shape))
        torch.manual_seed(4)
        s[0].bias = torch.nn.Parameter(torch.randn(s[0].bias.shape))
        torch.manual_seed(5)
        s[1].running_mean = torch.arange(4, dtype=torch.float)
        torch.manual_seed(6)
        s[1].running_var = torch.arange(4, dtype=torch.float) * 2 + 1
        self.layer = s
    def forward(self, x1):
        s1 = self.layer(x1)
        return s1
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)

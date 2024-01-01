
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s = torch.nn.Sequential(torch.nn.Conv2d(8, 4, 3, groups=2), torch.nn.BatchNorm2d(4, affine=False))
        s[0].weight = torch.nn.Parameter(torch.randn(s[0].weight.shape))
        s[1].weight = torch.nn.Parameter(torch.ones(s[1].weight.shape))
        s[1].bias = torch.nn.Parameter(torch.ones(s[1].bias.shape))
        self.layer = s
    def forward(self, x1):
        s1 = self.layer(x1)
        return torch.sum(s1)
x1 = torch.randint(0, 6, size=(1, 8, 4, 4))

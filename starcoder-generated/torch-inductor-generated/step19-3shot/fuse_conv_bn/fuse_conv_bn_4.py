
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 2, bias=False), torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 4, 2, bias=False))
        torch.manual_seed(3)
        s[0].weight = torch.nn.Parameter(torch.randn(s[0].weight.shape))
        torch.manual_seed(4)
        s[2].weight = torch.nn.Parameter(torch.randn(s[2].weight.shape))
        self.layer = s
    def forward(self, x1):
        s1 = self.layer(x1)
        return s1
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)

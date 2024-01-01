
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 3, bias=False)
        torch.manual_seed(2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.bias = torch.nn.Parameter(torch.randn(1))
        self.linear = torch.nn.Linear(1, 2, bias=False)
        self.bn.weight = torch.nn.Parameter(torch.zeros(1))
        self.bn.bias = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        s = self.conv(x)
        b = self.bn(s)
        x = b + b
        x = b + self.bias
        x = self.linear(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 1, 1)

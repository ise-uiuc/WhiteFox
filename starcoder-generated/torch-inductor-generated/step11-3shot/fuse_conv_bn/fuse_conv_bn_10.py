
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 12, 3)
        self.conv2 = torch.nn.Conv2d(24, 12, 1)
        self.bn = torch.nn.BatchNorm2d(12)

        torch.manual_seed(9)
        self.conv1.weight = torch.nn.Parameter(torch.randn(self.conv1.weight.shape))
        self.conv2.weight = torch.nn.Parameter(torch.randn(self.conv2.weight.shape))
        torch.manual_seed(10)
        self.conv1.bias = torch.nn.Parameter(torch.randn(self.conv1.bias.shape))
        self.conv2.bias = torch.nn.Parameter(torch.randn(self.conv2.bias.shape))
    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        torch.manual_seed(11)
        out2 = torch.nn.functional.pad(out1, (23, 7, 20, 2, 3, 1))
        torch.manual_seed(12)
        y = self.conv2(out2)
        return self.bn(y)
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)
x2 = torch.randn(1, 24, 6, 6)

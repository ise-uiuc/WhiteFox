
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(3, 8, 3, groups=4)
        torch.manual_seed(2)
        c.weight = torch.nn.Parameter(torch.randn(8, 3 // 4, 3, 3))
        torch.manual_seed(3)
        c.bias = torch.nn.Parameter(torch.randn(8))
        torch.manual_seed(5)
        self.bn0 = torch.nn.BatchNorm2d(8)
    def forward(self, x3):
        x3 = self.conv2(x3)
        x4 = self.bn0(x3) # This isn't fused
        return x4
# Inputs to the model
x3 = torch.randn(4, 3, 32, 32)

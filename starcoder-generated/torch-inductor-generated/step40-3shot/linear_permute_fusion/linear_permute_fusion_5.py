
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2).cuda()
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.conv2d(v4, self.conv.weight, bias=self.conv.bias, stride=2, padding=2)
        v2 = v1.permute(0, 2, 1, 3).cuda()
        v3 = self.bn(v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 10, 20, device='cuda')

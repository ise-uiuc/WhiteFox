
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        a = torch.nn.Conv2d(2, 3, 1)
        torch.manual_seed(3)
        a.weight = torch.nn.Parameter(torch.randn(a.weight.shape))
        self.conv = torch.nn.Conv2d(2, 3, 1)
        torch.manual_seed(7)
        self.conv.weight = torch.nn.Parameter(torch.randn(self.conv.weight.shape))
        self.fc = torch.nn.Linear(32, 10)
        torch.manual_seed(8)
        self.fc.weight = torch.nn.Parameter(torch.randn(self.fc.weight.shape))
        torch.manual_seed(9)
        self.fc.bias = torch.nn.Parameter(torch.randn(self.fc.bias.shape))
    def forward(self, x1):
        v0 = F.interpolate(self.conv(x1), mode="linear", align_corners=False)
        v1 = F.avg_pool2d(v0, kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True)
        v2 = torch.flatten(v1, 1)
        v3 = self.fc(v2)
    def forward(self, x1):
        v0 = F.interpolate(self.conv(x1), mode="linear", align_corners=False)
        v1 = F.avg_pool2d(v0, kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True)
        ret = self.fc(v1)
        return ret
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(11, 2, kernel_size=1)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x1):
        s_1 = self.conv(x1)
        s_2 = self.bn(s_1)
        t = torch.nn.functional.relu6(s_2)
        return t + torch.abs(t)

# Inputs to the model
x1 = torch.randn(1, 11, 1, 5, 5)

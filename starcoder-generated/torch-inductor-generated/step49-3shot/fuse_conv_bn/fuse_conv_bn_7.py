
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.conv1 = torch.nn.Conv3d(3, 3, 3)
        torch.manual_seed(3)
        self.bn = torch.nn.BatchNorm3d(3)
        self.conv2 = torch.nn.Conv3d(3, 3, 2)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.bn(s)
        y = self.conv2(t)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6, 6)

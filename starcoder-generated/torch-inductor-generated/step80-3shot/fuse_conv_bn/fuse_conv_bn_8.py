
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm3d(3)
        torch.manual_seed(3)
        self.conv2 = torch.nn.Conv3d(3, 3, 3)
        torch.manual_seed(3)
        self.conv3 = torch.nn.Conv3d(3, 3, 3)
        torch.manual_seed(3)
        self.soft_max = torch.nn.Softmax()
    def forward(self, x1):
        s = self.conv1(x1)
        y = self.bn(s)
        t = self.conv2(y)
        u = self.conv3(t)
        v = self.soft_max(u)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6, 6)

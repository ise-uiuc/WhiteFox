
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv45 = torch.nn.Sequential(torch.nn.ReLU(True), torch.nn.Conv2d(1, 2, 4), torch.nn.Conv2d(2, 4, 4))
        torch.manual_seed(1)
        self.conv34 = torch.nn.Sequential(torch.nn.ReLU(True), torch.nn.Conv2d(1, 2, 3), torch.nn.Conv2d(2, 4, 3))
        self.conv = torch.nn.ReLU(False)
        torch.manual_seed(1)
        self.bn789 = torch.nn.Sequential(torch.nn.BatchNorm2d(8), torch.nn.BatchNorm2d(8), torch.nn.BatchNorm2d(2))
        torch.manual_seed(1)
        self.bn67 = torch.nn.Sequential(torch.nn.ReLU(True), torch.nn.BatchNorm2d(8), torch.nn.BatchNorm2d(8))
    def forward(self, x1):
        s1 = self.conv(self.conv45(x1))
        s1 = self.conv(self.conv34(x1))
        s1 = self.bn789(s1)
        s1 = self.conv(s1)
        s1 = self.conv(self.bn67(s1))
        return s1 * s1
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)

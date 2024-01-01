
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv3d(4, 4, 1)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm3d(4)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(4, 4, 1)
        torch.manual_seed(1)
        self.bn2 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        x2  = self.conv1(x1)
        y1 = self.bn2(self.conv2(self.bn1(x2)))
        return y1
# Inputs to the model
x1 = torch.randn(2, 4, 8, 8, 8)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 3, 1)
        
        torch.manual_seed(2)
        self.bn1 = torch.nn.BatchNorm2d(3)

        torch.manual_seed(3)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=False)

        torch.manual_seed(4)
        self.bn3 = torch.nn.BatchNorm2d(3, track_running_stats=False)

        torch.manual_seed(5)
        self.bn4 = torch.nn.BatchNorm2d(3, affine=False, track_running_stats=False)

    def forward(self, x):

        y1 = self.conv(x)
        y2 = self.bn1(y1)
        y3 = self.bn2(y2)
        y4 = self.bn3(y3)
        y5 = self.bn4(y4)

        return y5

# Inputs to the model
x = torch.randn(1, 3, 10, 10)

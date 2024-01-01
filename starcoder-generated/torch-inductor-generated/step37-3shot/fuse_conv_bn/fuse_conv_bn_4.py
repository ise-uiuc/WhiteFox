
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bn = torch.nn.BatchNorm2d
        self.conv1 = nn.Conv2d(2, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 1)
        self.bn = bn(2)
        
        # bn(3) would be an invalid example since it's not tracking running stats
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.bn(x2)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)

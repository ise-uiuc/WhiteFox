
class Model(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, n_classes):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, middle_channels, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(middle_channels)
        self.conv2 = torch.nn.Conv2d(middle_channels, n_classes, 3, 1, 1)
 
    def forward(self, x1):
        t1 = self.bn1(F.gelu(self.conv1(x1)))
        t2 = self.conv2(t1)
        return t2

# Initializing the model
m = Model(3, 20, 10)

# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)

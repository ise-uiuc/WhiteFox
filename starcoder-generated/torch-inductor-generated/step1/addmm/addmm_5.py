
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1)
 
    def forward(self, *input):
        x = input
        x1 = self.conv1(x)
        x2 = self.bn(x)
        x3 = x1 + x2
        x4 = F.relu(x3)
        x5 = self.conv2(x4)
        x6 = x5 + x
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

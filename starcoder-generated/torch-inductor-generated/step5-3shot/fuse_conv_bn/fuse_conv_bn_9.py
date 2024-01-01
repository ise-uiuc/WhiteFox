
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(12)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        torch.manual_seed(12)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Inputs to the model
x = torch.randn(2, 3, 4, 4)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), groups=3, bias=False)
        self.conv1 = torch.nn.Conv2d(4, 3, kernel_size=(7, 7), groups=4, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), groups=3, bias=False) # Conv2d(3, 3, kernel_size=(7, 7) does NOT trigger the optimization, and the optimization is not applied.
        self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), groups=3, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# Inputs to the model
x = torch.randn(3, 4, 224, 224)

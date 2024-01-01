
torch.manual_seed(3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 3)
        self.activation1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 16, 3)
        self.activation2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(16, 32, 3)
        self.activation3 = torch.nn.ReLU()
        self.avgPool = torch.nn.AvgPool2d(3)
    def forward(self, x):
        s = self.conv1(x)
        s = self.activation1(s)
        s = self.conv2(s)
        s = self.activation2(s)
        s = self.conv3(s)
        s = self.activation3(s)
        y = self.avgPool(s)
        return y
# Inputs to the model
x = torch.randn(1, 4, 4, 4)

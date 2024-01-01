
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=3, bias=False) # Change the values of the parameters
        self.conv2 = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x2):
        x2 = self.conv1(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.conv2(x2)
        return x2
# Inputs to the model
x2 = torch.randn(1, 3, 5, 5)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(32, 32, 1)
        torch.manual_seed(0)
        self.conv2 = torch.nn.Conv2d(32, 32, 1)
        torch.manual_seed(0)
        self.b1 = torch.nn.BatchNorm2d(32)
        torch.manual_seed(0)
        self.b2 = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.b2(self.b1(x))
        return x
# Inputs to the model
x = torch.randn(1, 32, 32, 32)

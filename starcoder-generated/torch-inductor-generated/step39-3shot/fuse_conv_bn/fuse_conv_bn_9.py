
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=(4, 4), stride=(4, 4))
        self.bn0 = torch.nn.BatchNorm2d(2)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.bn = torch.nn.BatchNorm2d(2)
        self.output = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.bn1(x)
        x = self.bn(x)
        return self.output(x)
# Inputs to the model
x = torch.randn(1, 2, 4, 4)

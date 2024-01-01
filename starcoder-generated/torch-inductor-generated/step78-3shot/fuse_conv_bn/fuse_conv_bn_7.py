
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=(1, 1), dilation=(1, 1))
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        torch.manual_seed(1)
        self.relu1 = torch.nn.ReLU()

        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(32, 32, (3, 3), stride=(2, 2), bias=False, padding=(1, 1), dilation=(1, 1))
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(32, affine=False)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.bn1(x2)
        x2 = self.relu1(x2)
        x3 = self.conv(x2)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)

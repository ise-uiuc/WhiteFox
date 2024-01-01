
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # model components
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 5, 3, 1, 1)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(5, 5, 3, 2, 1)
        torch.manual_seed(1)
        self.conv3 = torch.nn.Conv2d(5, 5, 1, 2)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        torch.manual_seed(1)
        self.bn2 = torch.nn.BatchNorm2d(5)
        torch.manual_seed(1)
        self.bn3 = torch.nn.BatchNorm2d(5)

    def forward(self, x9):
        x3 = self.conv1(x9)
        x4 = self.bn1(x3)
        x5 = self.activation(x4)
        x6 = self.conv2(x5)
        x7 = self.bn2(x6)
        x8 = self.activation(x7)
        x9 = self.conv3(x8)
        x10 = self.bn3(x9)
        return x10
# Inputs to the model
x9 = torch.randn(1, 3, 32, 32)

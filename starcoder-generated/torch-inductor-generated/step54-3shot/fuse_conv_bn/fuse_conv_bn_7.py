
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(1, 1, 2, bias=False)
        torch.manual_seed(0)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.functional.relu
        self.conv2 = torch.nn.Conv2d(1, 1, 2, bias=False)
        self.sigmoid = torch.sigmoid
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)

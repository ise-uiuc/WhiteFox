
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.conv1 = torch.nn.Conv2d(2, 3, 1, bias=False)
        torch.manual_seed(2)
        self.bn = torch.nn.BatchNorm1d(3)
        self.conv2 = torch.nn.Conv2d(3, 2, 2, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(F.relu(x))
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)

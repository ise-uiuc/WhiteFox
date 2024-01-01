
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 5)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x6):
        v = self.relu(self.bn(self.conv(x6)))
        return v
# Inputs to the model
x6 = torch.randn(1, 3, 12, 12)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1):
        out1 = self.bn1(self.relu(self.conv1(x1)))
        out2 = torch.sigmoid(out1)
        return out2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

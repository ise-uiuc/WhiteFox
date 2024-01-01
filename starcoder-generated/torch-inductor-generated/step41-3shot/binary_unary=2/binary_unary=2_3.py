
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout2d(0.25)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        t1 = self.conv2(v2)
        v3 = self.dropout(t1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

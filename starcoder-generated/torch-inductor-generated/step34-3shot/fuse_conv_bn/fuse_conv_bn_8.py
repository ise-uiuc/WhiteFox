
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 4)
        self.batchnorm2d = torch.nn.BatchNorm2d(3)
        self.dropout = torch.nn.Dropout()

    def forward(self, x1, x2):
        s = torch.cat([x1, x2], dim=1)
        s = self.conv1(s)
        s = self.batchnorm2d(s)
        s = self.dropout(s)
        return s
# Inputs to the model
x1 = torch.rand(1, 3, 6, 6)
x2 = torch.rand(1, 3, 6, 6)

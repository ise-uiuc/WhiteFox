
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 15, 2, stride=2)
        self.mp = torch.nn.MaxPool2d(2, stride=2)
        self.dropout = torch.nn.Dropout2d()
        self.conv2 = torch.nn.Conv2d(15, 35, 1, stride=1)
        self.b1 = torch.nn.BatchNorm2d(35)
        self.b2 = torch.nn.BatchNorm2d(35)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.mp(t1)
        t3 = self.dropout(t2)
        t4 = self.conv2(t3)
        t5 = self.b1(t4)
        t6 = self.b2(t4)
        t7 = t5 + t6
        return t7
# Inputs to the model
x1 = torch.randn(5, 4, 28, 28)

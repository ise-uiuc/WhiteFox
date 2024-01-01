
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=2)
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=2)
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.pool1(x1)
        t2 = self.pool2(x1)
        t3 = self.conv1(x1)
        t4 = t1 + t2 + t3
        t5 = torch.relu(t4)
        return t5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

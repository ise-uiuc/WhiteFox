
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(8, 8, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.out = torch.nn.Linear(10, 4)
    def forward(self, x1):
        v1 = self.pool1(torch.sigmoid(self.conv1(x1)))
        v2 = self.pool2(torch.sigmoid(self.conv2(v1)))
        v3 = v2.view(-1, 4, 24)
        v4 = torch.sigmoid(self.out(v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16,16,(65, 65),stride=1,padding=0)
        self.pool1 = torch.nn.MaxPool2d((33, 33),stride=1,padding=0)
        self.conv2 = torch.nn.Conv2d(16,32,(4, 4),stride=1,padding=0)
        self.pool2 = torch.nn.MaxPool2d((2, 2),stride=2,padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.pool1(v2)
        v4 = v3
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.pool2(v6)
        return v7
# Inputs to the model
torch.manual_seed(42)
x1 = torch.rand((1,16,64,64))

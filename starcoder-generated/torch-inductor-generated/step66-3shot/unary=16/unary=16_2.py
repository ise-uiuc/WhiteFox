
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input channels = 3, output channels = 32, kernel size = 3x3
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear1 = torch.nn.Linear(4 * 4 * 32, 256)
        self.linear2 = torch.nn.Linear(256, 10)
 
    def forward(self, x1):
        # apply two convolutions followed by batch normalization and
        # two relu activations and max pooling
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        v3 = F.relu(v2)
        v4 = F.relu(self.conv2(v3))
        v5 = F.max_pool2d(v4, 2)
        # we reshape after the convolutions/activations to avoid flattening
        v6 = v5.view(-1, 4 * 4 * 32)
        v7 = self.linear1(v6)
        v8 = self.relu(v7)
        v9 = self.dropout(v8)
        return self.linear2(v9)

# Initializing the model
m = Model()

# Inputs to the model. By default, random data will be generated with the same shape as defined in the Pytorch function __init__
x1 = torch.randn(2, 3, 32, 32)

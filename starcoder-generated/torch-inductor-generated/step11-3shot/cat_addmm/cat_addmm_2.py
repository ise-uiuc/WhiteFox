
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels * 2, in_channels)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
         
    def forward(self, x1):
        v1 = F.relu(self.linear(x1))
        v2 = self.pool(v1)
        return v2

class Model(nn.Module):
    def __init__(self, block=ConvLayer):
        super(Model, self).__init__()
        self.block = block
        self.conv1 = block(75, 30)
        self.conv2 = block(30, 40)
        self.conv3 = block(40, 50)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return torch.cat([v1, v2, v3])

# Initializing the model
m2 = Model()

# Inputs to the model
x = torch.randn(64, 75)

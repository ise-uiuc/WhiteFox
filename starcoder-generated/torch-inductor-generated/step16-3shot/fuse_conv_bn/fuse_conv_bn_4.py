

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv3 = nn.Conv2d(3, 3, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x

# Note: The second conv2d has a different number of inputs.  
# Note: The model has two output paths in forward().  
# Note: One path is not returning a value.  
# Input to the model
x = torch.randn(1, 3, 10, 10)

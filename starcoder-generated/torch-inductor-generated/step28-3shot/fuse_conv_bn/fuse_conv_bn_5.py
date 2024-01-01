
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 2)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 4, 2)
        self.bn2 = torch.nn.BatchNorm2d(4)
        self.conv3 = torch.nn.Conv2d(4, 3, 2)
        self.relu = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(3, 3, 2)
        self.bn3 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.conv2(x2) 
        x4 = self.bn2(x3) 
        x5 = self.conv3(x4)
        x6 = self.relu(x5)
        x7 = self.conv4(x6)  
        x8 = self.bn3(x7)
        return x8
# Inputs to the model
x = torch.randn(1, 4, 8, 8)


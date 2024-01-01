
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=13, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 3, stride=1, padding=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=9, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 2, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.pool1(v1) # Pooling kernel size: 3 x 3
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.pool2(v4) # Pooling kernel size 2 x 2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7.view(1, -1)
        return v8
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

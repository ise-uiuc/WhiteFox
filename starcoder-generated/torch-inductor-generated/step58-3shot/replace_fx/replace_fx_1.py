.
# This is a typical model for image classification, using Conv2d, BatchNorm2d, AvgPool2d, and Linear layers.
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(self.conv1.out_channels,
                                           momentum=0.01, eps=0.001)
        self.bn2 = torch.nn.BatchNorm2d(self.conv2.out_channels,
                                           momentum=0.01, eps=0.001)
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4*4*50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.avg_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.bn2(x)
        x = self.avg_pool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output
# Inputs to the model.
x = torch.randn(1, 1, 32, 32)

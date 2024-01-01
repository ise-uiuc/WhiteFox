
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(16, affine=False)
        self.conv1 = torch.nn.Conv2d(16, 16, 2, dilation=2)
        self.bn2 = torch.nn.BatchNorm1d(16, affine=False)
        self.conv2 = torch.nn.Conv2d(16, 16, 2, dilation=2)
        self.bn3 = torch.nn.BatchNorm1d(16, affine=False)
        self.conv3 = torch.nn.Conv2d(16, 16, 2, dilation=2)
        self.bn4 = torch.nn.BatchNorm1d(16, affine=False)
        self.conv4 = torch.nn.Conv2d(16, 16, 2, dilation=2)
        self.bn5 = torch.nn.BatchNorm1d(16, affine=False)
        self.conv5 = torch.nn.Conv2d(16, 16, 2, dilation=2)
        self.maxpool1 = torch.nn.MaxPool2d(2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(16*5*5, 2)
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.conv4(x)
        x = self.bn5(x)
        x = self.conv5(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 12, 12)

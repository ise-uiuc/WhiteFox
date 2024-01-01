
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=200,  kernel_size=(1, 1), stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=200, out_channels=255, kernel_size=(1, 1), stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=255, out_channels=64,  kernel_size=(1, 1), stride=1, padding=0)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=7, stride=7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.pool1(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = self.pool2(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = self.pool3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 119, 119)

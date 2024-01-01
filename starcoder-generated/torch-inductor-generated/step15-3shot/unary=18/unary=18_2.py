
class model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu6 = nn.ReLU()
        # 10 for CIFAR10 and 3 for CIFAR 100
        self.linear = nn.Linear(out_channels, 10)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.relu3(v5)
        v7 = self.conv4(v6)
        v8 = self.relu4(v7)
        v9 = self.conv5(v8)
        v10 = self.relu5(v9)
        v11 = self.conv6(v10)
        v12 = self.relu6(v11)
        v13 = v12.view(v12.size(0), -1)
        v14 = self.linear(v13)
        return v14
in_channels = 4
out_channels = 16
padding = 1
kernel_size = 3
# Inputs to the model
x = torch.randn(1, 4, 32, 32)

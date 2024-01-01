
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 3)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 32)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = (2, 2), padding = 1)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 32)
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = (2, 2), padding = 1)
        self.batchnorm3 = torch.nn.BatchNorm2d(num_features = 32)
        self.conv4 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = (2, 2), padding = 1)
        self.batchnorm4 = torch.nn.BatchNorm2d(num_features = 32)
        self.conv5 = torch.nn.Conv2d(in_channels = 32, out_channels = 256, kernel_size = 3, padding = 0)
        self.batchnorm5 = torch.nn.BatchNorm2d(num_features = 256)
    def forward(self, x):
        c1 = torch.nn.functional.relu(self.batchnorm1(self.conv1(x)))
        c2 = torch.nn.functional.relu(self.batchnorm2(self.conv2(c1)))
        c3 = torch.nn.functional.relu(self.batchnorm3(self.conv3(c2)))
        c4 = torch.nn.functional.relu(self.batchnorm4(self.conv4(c3)))
        r = self.batchnorm5(self.conv5(c4))
        return r
# Inputs to the model
x = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv1d(in_channels=10, out_channels=12, kernel_size=3)
        self.bn1a = torch.nn.BatchNorm1d(num_features=12)
        self.conv1b = torch.nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3)
        self.bn1b = torch.nn.BatchNorm2d(num_features=12)
        self.conv2a = torch.nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3)
        self.bn2a = torch.nn.BatchNorm2d(num_features=14)
        self.conv2b = torch.nn.Conv3d(in_channels=24, out_channels=26, kernel_size=3)
        self.bn2b = torch.nn.BatchNorm3d(num_features=26)
        self.conv3a = torch.nn.Conv3d(in_channels=14, out_channels=28, kernel_size=3)
        self.bn3a = torch.nn.BatchNorm3d(num_features=28)
        self.conv3b = torch.nn.Conv3d(in_channels=10, out_channels=28, kernel_size=3)
        self.bn3b = torch.nn.BatchNorm3d(num_features=28)
    def forward(self, x):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.conv1b(x)
        x = self.avg_pool(x)
        x = self.bn1b(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.conv2b(x)
        x = self.avg_pool3d(x)
        x = self.bn2b(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.conv3b(x)
        x = self.avg_pool3d(x)
        x = self.bn3b(x)
        return x
# Inputs to the model
x = torch.randn(3, 10, 12)

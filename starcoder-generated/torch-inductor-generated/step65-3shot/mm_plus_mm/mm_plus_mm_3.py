
# Add batch normalization layers in the constructor function (not in forward).
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=6)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=12)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.batch_norm4 = nn.BatchNorm2d(num_features=12)
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = F.leaky_relu(out)
        return out

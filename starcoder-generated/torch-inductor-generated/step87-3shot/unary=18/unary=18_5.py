
class Model(nn.Module):
    def __init__(self):
        super(DeepV2, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=13)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_1 = self.conv1_1(x1)
        x1_2 = self.conv1_2(x1_1)
        x2 = self.conv2(x1_2)
        x2_1 = self.conv2_1(x2)
        x3 = torch.sigmoid(x2_1)
        x4 = torch.sigmoid(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1,3,400,700)

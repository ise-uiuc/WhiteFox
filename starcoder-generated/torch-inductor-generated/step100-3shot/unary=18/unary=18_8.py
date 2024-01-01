
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=80, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(in_channels=80, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x1):
        v1 = self.conv4(torch.sigmoid(self.conv3(torch.sigmoid(self.conv2(torch.sigmoid(self.conv1(x1)))))))
        print(v1.shape)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)

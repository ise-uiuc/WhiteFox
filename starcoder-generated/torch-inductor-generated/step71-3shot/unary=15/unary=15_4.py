
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(2, 2), padding=(1, 2), dilation=(1, 1), groups=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1)
        self.conv4 = nn.Conv2d(2, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1)
        self.conv5 = nn.Conv2d(2, 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1)
        self.conv6 = nn.Conv2d(2, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.maxpool(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = self.conv6(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)

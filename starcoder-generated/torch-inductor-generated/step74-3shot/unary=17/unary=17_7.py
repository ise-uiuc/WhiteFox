
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), dilation=(1, 1))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        self.relu_1 = torch.nn.ReLU(inplace=False)
        self.conv_2 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0), dilation=(1, 1))
        self.relu_2 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.avgpool(v1)
        v3 = self.relu_1(v2)
        v4 = self.conv_2(v3)
        v5 = self.relu_2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)

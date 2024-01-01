
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d_13 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2d_6 = torch.nn.Conv2d(19, 256, 1, stride=1, padding=0, bias=False)
        self.avg_pool2d_9 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(256, 256, 4, stride=1, padding=0, bias=False)
    def forward(self, x4):
        v11 = self.maxpool2d_13(x4)
        v12 = self.conv2d_6(v11)
        v13 = torch.transpose(v12, 3, 1)
        v14 = self.avg_pool2d_9(v13)
        v15 = torch.transpose(v14, 2, 3)
        v16 = self.conv_transpose_2(v15)
        v17 = torch.sigmoid(v16)
        v4 = v16 * v17
        return v4
# Inputs to the model
x4 = torch.randn(1, 19, 56, 56)

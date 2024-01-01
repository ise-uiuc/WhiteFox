
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        # self.conv2 = torch.nn.Conv2d(3, 7, kernel_size = 17, kernel_size=(17, 23), stride=(3, 1), dilation=(4, 2), groups = 4, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

        # self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=(3, 7))
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((8, 5))

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = torch.sigmoid(x1)
        x = self.avg_pool2d(x1)
        return x
# Inputs to the model
x = torch.randn(1, 3, 28, 28)

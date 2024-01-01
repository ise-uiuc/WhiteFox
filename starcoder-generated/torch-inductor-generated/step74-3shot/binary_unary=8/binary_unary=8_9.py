
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool18 = torch.nn.AvgPool2d(kernel_size=9, stride=9, padding=9)
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.avg_pool18(x1)
        v1_reshape = v1.reshape(1, -1)
        v2 = self.conv1(v1_reshape)
        v2_reshape = v2.reshape(1, -1)
        v3 = self.conv2(v2_reshape)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

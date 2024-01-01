
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(1, 9), stride=(1, 4), padding=(0, 4), dilation=2, groups=2, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 64, kernel_size=(1, 9), stride=(1, 4), padding=(0, 2), dilation=2, groups=2, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(x1)
        v4 = torch.sigmoid(v3)
        return v2, v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)

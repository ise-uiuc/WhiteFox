
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=2)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return torch.sigmoid(v5)
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)

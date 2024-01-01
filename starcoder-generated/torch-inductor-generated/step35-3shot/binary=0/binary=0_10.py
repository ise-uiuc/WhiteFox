
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=(5, 1), stride=(1, 2), padding=(1))
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, 3), stride=(2, 1), padding=(2, 1))
        self.conv3 = torch.nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(2, 3), padding=(3, 1))
        self.conv4 = torch.nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2))
    def forward(self, x1):
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

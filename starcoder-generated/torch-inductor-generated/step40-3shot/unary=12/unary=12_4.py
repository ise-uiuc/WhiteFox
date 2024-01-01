
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(32, 256, kernel_size=(12, 4), stride=(1, 1), padding=(6, 2), dilation=(2, 1))
        self.sigmoid2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid1(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.sigmoid2(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 128, 864, 160)

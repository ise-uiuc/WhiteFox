
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, (3, 8), stride=(1, 2), padding=(1, 3), dilation=(1, 1))
        self.conv = self.conv0
        self.conv1 = torch.nn.Conv2d(8, 16, (1, 3), stride=(1, 2), padding=0, dilation=(1, 2))
        self.conv2 = torch.nn.Conv2d(16, 32, (4, 1), stride=(2, 1), padding=(2, 1), dilation=(2, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

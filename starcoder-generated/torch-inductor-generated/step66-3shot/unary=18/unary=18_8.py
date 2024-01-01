
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=11, out_channels=40, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_2 = torch.nn.Conv2d(in_channels=40, out_channels=7, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(3, 3))
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 23, 18)

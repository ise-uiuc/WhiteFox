
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_conv = torch.nn.Sequential(torch.nn.Conv2d(64, 16, kernel_size=(35, 1), stride=(1, 1), padding=(33, 0)), torch.nn.Conv2d(64, 16, kernel_size=(1, 35), stride=(1, 1), padding=(0, 33)))
    def forward(self, x1):
        v1 = self.conv_conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 392, 392)

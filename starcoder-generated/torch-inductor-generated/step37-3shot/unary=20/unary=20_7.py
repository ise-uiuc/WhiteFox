
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=(13, 1), stride=(1, 1), padding=(0, 0), groups=4, bias=True)
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=5, kernel_size=(11, 2), stride=(12, 21), padding=(4, 4), dilation=8, groups=6, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=(1, 2, 3), stride=(1, 2), padding=(3, 4), bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 52, 52)

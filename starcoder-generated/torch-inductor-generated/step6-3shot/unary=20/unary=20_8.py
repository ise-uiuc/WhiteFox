
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d0 = torch.nn.Conv2d(64, 8, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=(2, 2))
        self.conv2d1 = torch.nn.Conv2d(8, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.pad = torch.nn.ConstantPad2d(1, 0.0)
    def forward(self, x1):
        v0 = self.conv2d0(x1)
        v1 = self.conv2d1(v0)
        v2 = self.pad(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)

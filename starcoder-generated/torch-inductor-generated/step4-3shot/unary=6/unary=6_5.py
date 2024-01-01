
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convpool1 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=[15, 1], padding=[9, 0], stride=[16, 1], groups=1), torch.nn.MaxPool2d(kernel_size=5, stride=4, padding=0, dilation=1, ceil_mode=False))
        self.convpool1_out_ch = 13
    def forward(self, x1):
        v1 = self.convpool1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 64, 8192, 2)*5

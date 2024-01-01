
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        self.conv2 = torch.nn.ConvTranspose2d(4, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(1, 1), groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28) # The first batch size dimension has to be 1: https://github.com/pytorch/pytorch/issues/59919

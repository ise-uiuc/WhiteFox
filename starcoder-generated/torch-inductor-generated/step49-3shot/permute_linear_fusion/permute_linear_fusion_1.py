
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    def forward(self, x1):
        v1 = self.conv2d_0(x1)
        return x1 + v1[:, :, 1, 1]
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)

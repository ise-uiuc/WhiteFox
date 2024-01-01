
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0,), output_padding=(0,), groups=1, dilation=(1,), bias=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return self.sigmoid(v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)

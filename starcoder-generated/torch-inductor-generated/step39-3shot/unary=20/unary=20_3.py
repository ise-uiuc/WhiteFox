
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(1, 1), groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 4, 4)

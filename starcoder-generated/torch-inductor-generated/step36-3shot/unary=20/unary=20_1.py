
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=7, out_channels=5, kernel_size=5, stride=5, padding=5, bias=True)
    def forward(self, x1):
        t2 = self.conv_transpose(x1)
        t3 = torch.sigmoid(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 7, 640, 480)

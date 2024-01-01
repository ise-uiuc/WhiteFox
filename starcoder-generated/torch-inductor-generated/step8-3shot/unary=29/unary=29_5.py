
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(in_channels=8, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 8, 1, stride=1, padding=0)
        torch.manual_seed(12345)
    def forward(self, x1):
        e1 = self.conv(x1)
        e2 = self.conv_transpose1(e1)
        return e2
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)

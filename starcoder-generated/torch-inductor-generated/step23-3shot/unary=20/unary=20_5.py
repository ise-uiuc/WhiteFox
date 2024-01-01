
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1d = torch.nn.ConvTranspose1d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, groups=1, bias=True
        )
    def forward(self, t1):
        t2 = self.conv_transpose1d(t1)
        t3 = torch.sigmoid(t2)
        return t3
# Input to the model
t1 = torch.randn(1, 1024, 128)

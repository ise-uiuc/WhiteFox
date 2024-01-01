
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(512, 64, 2, 2, bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1), out_channels=512)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.clamp(v2, 0.0, 2.0)
        v4 = v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 512, 1, 1)

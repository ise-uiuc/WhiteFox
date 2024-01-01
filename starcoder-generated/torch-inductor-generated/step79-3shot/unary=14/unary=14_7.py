
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v5 = self.conv_transpose2(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 3, 3)

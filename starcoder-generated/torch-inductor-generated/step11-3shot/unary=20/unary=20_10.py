
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.modules.conv.ConvTranspose2d(in_channels=39, out_channels=64, kernel_size=(1, 11), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.conv_transpose2 = torch.nn.modules.conv.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1, 9), stride=(8, 1), padding=(0, 0), output_padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 39, 1, 189)

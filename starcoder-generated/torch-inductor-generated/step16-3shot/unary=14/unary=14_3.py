
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(in_channels=3,
                                                     out_channels=16,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1)
        torch.nn.init.constant_(self.conv_transpose_1.weight, 0.0)
        torch.nn.init.constant_(self.conv_transpose_1.bias, 0.0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
torch.manual_seed(1)
x1 = torch.randn(1, 3, 64, 64)

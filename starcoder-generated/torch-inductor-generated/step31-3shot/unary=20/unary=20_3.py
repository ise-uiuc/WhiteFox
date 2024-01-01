
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=6)
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=(3, 3), stride=(1, 2), padding=(2, 1), groups=1)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)

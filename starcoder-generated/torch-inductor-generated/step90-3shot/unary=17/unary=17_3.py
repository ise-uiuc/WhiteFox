
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(16, 16, 3, stride=(1, 1, 1), padding=(1, 1, 1), output_padding=(0, 0, 0))
        self.conv = torch.nn.Conv3d(16, 64, 5, stride=(1, 1, 1), padding=(2, 2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 101, 101, 101)

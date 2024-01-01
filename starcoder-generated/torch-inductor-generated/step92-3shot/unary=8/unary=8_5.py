
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_transpose = torch.nn.ConvTranspose2d(in_channels=6, out_channels=2, kernel_size=3, padding=1)
        self.conv2_transpose = torch.nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv3_transpose = torch.nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3)
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = self.conv1_transpose(v1)
        v3 = self.conv2_transpose(v2)
        v4 = v3.flip(2)
        v5 = v4 * (1/8)
        v6 = self.conv3_transpose(v5)
        v7 = torch.sigmoid(v6)
        v8 = v7 * (1/8)
        v9 = v1 * v8
        v10 = v9 / 1
        return v10
# Inputs to the model
x1 = torch.randn(1, 6, 10, 10)

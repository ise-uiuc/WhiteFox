
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 256, 1, stride=1, padding=0)
        self.conv_transpose_1_1 = torch.nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0)
        self.conv_transpose_1_2 = torch.nn.ConvTranspose2d(256, 64, 3, stride=1, padding=1)
        self.conv_transpose_1_3 = torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv_transpose_1(x)
        v1_1 = self.conv_transpose_1_1(v1)
        v1_2 = self.conv_transpose_1_2(v1_1)
        v1_3 = self.conv_transpose_1_3(v1_2)
        v3 = torch.sigmoid(v1_3)
        v5 = v1_3 * v3
        return v5
# Inputs to the model
x = torch.randn(1, 3, 512, 512)

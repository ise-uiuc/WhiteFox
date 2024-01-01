
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 16, kernel_size=[3, 8], stride=[1, 1], padding=[6, 4])
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 40, 40)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 11, kernel_size=(1, 6), stride=(16, 2), groups=8, padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(16, 1, 12, 12)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(4, 8, kernel_size=(3, 2, 1), stride=(3, 2, 1), padding=(0, 1, 0), output_padding=(0, 1, 0))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 12, 22, 10)

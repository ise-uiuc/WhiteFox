
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(5, 3, kernel_size=(3, 5, 9), stride=(1, 2, 1), padding=(2, 1, 10))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(8, 5, 16, 36, 52)

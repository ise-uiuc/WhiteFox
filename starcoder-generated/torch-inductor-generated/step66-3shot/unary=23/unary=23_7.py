
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(4, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = torch.cat((x1, v2), 1)
        return v3
# Inputs to the model
x1 = torch.randn(2, 4, 24, 23, 22)
x2 = torch.randn(2, 2, 24, 23, 22)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = ConvTranspose2d(24, 94, (5, 5), stride=(5, 5), bias=True, dilation=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 24, 20, 30)

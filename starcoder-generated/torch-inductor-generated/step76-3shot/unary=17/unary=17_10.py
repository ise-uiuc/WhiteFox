
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(3, 3, (32))
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(3, 3, (32))
        self.conv_transpose_3 = torch.nn.ConvTranspose1d(3, 3, (32))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 70)

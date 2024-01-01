
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 4, 5, stride=3, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 4, 5, stride=(2, 1), padding=(3, 2))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 5, 3)

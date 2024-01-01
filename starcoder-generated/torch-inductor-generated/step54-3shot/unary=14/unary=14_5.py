
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(10, 10, 3, stride=1, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(10, 1, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = self.conv_transpose_3(v1)
        v3 = v2.view(-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)

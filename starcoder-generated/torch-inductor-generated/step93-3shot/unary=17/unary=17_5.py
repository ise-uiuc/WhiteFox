
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 128, (18, 50), padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 0, 50)

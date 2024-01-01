
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=4, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        x2 = v1.reshape(1)
        return x2
# Inputs to the model
x1 = torch.randn(16, 16, 4, 4)

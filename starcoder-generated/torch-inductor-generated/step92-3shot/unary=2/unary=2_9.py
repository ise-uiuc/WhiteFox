
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 8, kernel_size=3, stride=1, padding=4, output_padding=0)
    def forward(self, x1):
        return self.conv_transpose(x1)
# Inputs to the model
x1 = torch.randn(2, 6, 1, 1)

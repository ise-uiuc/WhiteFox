
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_transpose = torch.nn.ConvTranspose2d(3, 16, 5, stride=2, padding=1, output_padding=0, bias=False)
    def forward(self, x1):
        x2 = x1.repeat(1, 1, 3, 3)
        x3 = self.conv2d_transpose(x2)
        return x3
# Inputs to the model
x1 = torch.randn(2, 3, 8, 8)

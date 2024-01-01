
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 20, 2, stride=2)
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = x2+1
        return x3
# Inputs to the model
x1 = torch.randn(1, 10, 8, 8)

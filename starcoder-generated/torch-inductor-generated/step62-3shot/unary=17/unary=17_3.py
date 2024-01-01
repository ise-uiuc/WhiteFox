
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 10, (2, 2), bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 10, 3, 5)

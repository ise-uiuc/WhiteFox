
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=5, stride=5, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 14, 26)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 1, 1)
    def forward(self, x):
        x1 = self.conv_transpose(x)
        return x1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

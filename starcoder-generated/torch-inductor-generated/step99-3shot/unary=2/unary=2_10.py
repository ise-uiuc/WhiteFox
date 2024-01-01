
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 32, (1, 1), 2)
    def forward(self, x1):
        result = self.conv_transpose(x1) # (2, 32, 48, 64)
        return result
# Inputs to the model
x1 = torch.randn(1, 2, 48, 64)

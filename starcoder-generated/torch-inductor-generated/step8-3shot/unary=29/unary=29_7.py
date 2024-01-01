
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        return self.conv_transpose(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 127)

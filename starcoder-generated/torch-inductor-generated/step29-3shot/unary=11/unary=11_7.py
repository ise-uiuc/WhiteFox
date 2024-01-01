
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 2, 3, stride=2, padding=1)
    def forward(self):
        return self.conv_transpose(x1) + 3
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)

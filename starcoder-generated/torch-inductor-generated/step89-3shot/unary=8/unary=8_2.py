
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(31, 21, 2, stride=2, padding=2, output_padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(21, 27, 4, stride=1, padding=1, output_padding=0)
    def forward(self, x2):
    return
# Inputs to the model
x2 = torch.randn(1, 31, 32, 32)

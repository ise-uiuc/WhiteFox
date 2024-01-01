
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(3,  2, kernel_size=5, stride=1, padding=2, output_padding=1)
    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(3, 4, kernel_size=2, stride=4)
    def forward(self, x):
        x_2 = self.conv_transpose(x)
        return torch.sigmoid(x_2)
# Inputs to the model
x = torch.randn(1, 3, 16, 16)

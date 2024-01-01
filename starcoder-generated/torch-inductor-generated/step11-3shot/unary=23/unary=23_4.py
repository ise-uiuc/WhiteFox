
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(8, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 8, 256, 128, 128)

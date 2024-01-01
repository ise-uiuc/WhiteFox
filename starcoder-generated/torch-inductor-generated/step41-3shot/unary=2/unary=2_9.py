
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(3, 3))
    def forward(self, x1):
        return self.conv_transpose2(self.conv_transpose(x1))
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)

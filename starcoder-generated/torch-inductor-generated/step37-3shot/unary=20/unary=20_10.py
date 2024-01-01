
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tanh = torch.nn.Tanh()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 5, kernel_size=(2, 1), padding=(0, 0), bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 3, kernel_size=(1, 1), padding=(0, 0), bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 200, 750)

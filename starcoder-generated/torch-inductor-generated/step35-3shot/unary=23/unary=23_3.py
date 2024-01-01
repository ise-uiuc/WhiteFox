
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv       = torch.nn.Conv2d(96, 192, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv_transpose = torch.nn.ConvTranspose2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 96, 35, 192)

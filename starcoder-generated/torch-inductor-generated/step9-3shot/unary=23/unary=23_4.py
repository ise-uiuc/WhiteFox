
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(8, 27, kernel_size=(2, 2), stride=(2, 2), padding=(2, 2))
    def forward(self, x1):
        v1 = self.tconv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 10, 10)

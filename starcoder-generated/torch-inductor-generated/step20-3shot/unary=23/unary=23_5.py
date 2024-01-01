
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose1 = torch.nn.ConvTranspose3d(3, 3, kernel_size=(3, 5, 1), stride=(1, 3, 2), padding=(2, 1, 0))
    def forward(self, x1):
        v1 = self.convTranspose1(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 15, 32, 100)

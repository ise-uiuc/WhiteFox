
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling_transposedconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = torch.nn.MaxPool2d(kernel_size=2, stride=3, padding=1)(x1)
        v2 = self.pooling_transposedconv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)

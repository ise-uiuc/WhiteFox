
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tranpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv_tranpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 20, 20)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 2, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 16, 16)

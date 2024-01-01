
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(64, 3, 2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 64, 94)

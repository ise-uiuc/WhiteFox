
class Model(torch.nn.Module):
    def __init__(self, conv_transpose):
        super().__init__()
        self.conv_transpose = conv_transpose
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = x2 > 0
        x4 = x2 * 0.25
        x5 = torch.where(x3, x2, x4)
        return x5
# Inputs to the model
x1 = torch.randn(16, 3, 16, 16)

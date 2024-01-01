
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(1, 9, 3)
        self.conv = torch.nn.Conv2d(9, 5, 7)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = torch.selu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 129, 101)

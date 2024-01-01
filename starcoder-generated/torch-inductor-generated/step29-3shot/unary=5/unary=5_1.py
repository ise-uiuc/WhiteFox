
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 5, stride=2, padding=2, dilation=1)
        self.linear = torch.nn.Linear(4, 5)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = self.linear(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
x2 = torch.randn(1, 5, 1)

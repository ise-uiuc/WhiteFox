
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_transpose = torch.nn.ConvTranspose2d(5, 16, kernel_size=2, dilation=3, stride=3, padding=5)
    def forward(self, x1):
        v1 = self.conv1_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 111, 111)

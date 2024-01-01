
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 20, 6, stride=6, padding=0)
        self.conv2_transpose = torch.nn.ConvTranspose2d(20, 20, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2_transpose(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 1, 5)

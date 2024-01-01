
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 3, kernel_size=(2, 1), stride=1, padding=0)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=5, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 1, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 10, 8, stride=4, padding=1)
        self.conv2d = torch.nn.Conv2d(10, 7, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2d(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

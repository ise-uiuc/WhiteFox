
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 2, 3, 1, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 3, 1, 1)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.softmax(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 1, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(4, 1, 2, padding=0)
    def forward(self, x1):
        v1 = torch.tanh(self.conv1(x1))
        v2 = torch.tanh(self.conv_transpose(v1))
        v3 = torch.tanh(self.conv2(v2))
        v4 = torch.tanh(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)

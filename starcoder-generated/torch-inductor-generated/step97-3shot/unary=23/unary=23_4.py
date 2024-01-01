
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 25, 1)
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=18, stride=6, padding=12)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.max_pooling(v1)
        v3 = torch.tanh(v2 * 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 18, 6)

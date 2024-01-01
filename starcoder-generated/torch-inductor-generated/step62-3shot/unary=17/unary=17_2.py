
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 8, padding=3, padding_mode='reflect')
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose1d(64, 1, kernel_size=5, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

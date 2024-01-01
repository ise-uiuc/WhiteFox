
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=5, stride=2, padding=1)
    def forward(self, x1):
        v4 = self.conv2(self.conv1(self.conv(x1))) #conv(conv_transpose(conv_transpose(in)))
        v3 = torch.sigmoid(v4)
        v2 = v4 * v3 # conv(conv(in)) * sigmoid()
        v1 = torch.sigmoid(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)

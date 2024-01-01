
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.Activation('sigmoid')(v1)
        v3 = self.conv2(v2)
        v4 = nn.Sigmoid()(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)

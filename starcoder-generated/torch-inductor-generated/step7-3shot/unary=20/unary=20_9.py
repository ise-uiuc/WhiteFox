
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_5 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.deconv2d_1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=0)
    def forward(self, input):
        v2 = self.conv2d_5(input)
        v3 = self.deconv2d_1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
input = torch.randn(1, 64, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, padding=1, stride=1, bias=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 2, padding=0, stride=2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_transpose(self.conv(x1))
        v2 = self.sigmoid(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

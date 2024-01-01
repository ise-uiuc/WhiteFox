
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution2d5 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.convolution2d5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 224, 224)

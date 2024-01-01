
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, 1, 1)
        self.bn2d = torch.nn.BatchNorm2d(1)
    def forward(self, img):
        x = self.conv2d(img)
        x = self.bn2d(x)
        return x
# Inputs to the model
img = torch.randn(1, 1, 2, 2)

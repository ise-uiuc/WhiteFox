
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torchvision.transforms.Resize(x1.shape[2:] * 2)(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

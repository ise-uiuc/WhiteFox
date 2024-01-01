
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride = 2, padding=2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride = 2, padding=2)
        self.conv3 = torch.nn.ConvTranspose2d(50, 20, 5, stride = 2, padding=2)
        self.conv4 = torch.nn.ConvTranspose2d(20, 1, 5, stride = 2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(280, 1, 28, 28)

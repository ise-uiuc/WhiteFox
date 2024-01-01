
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 1, 2)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 3)
        self.conv3 = torch.nn.ConvTranspose2d(1, 1, 4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, (4, 4), stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(6, 6, (2, 2), stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(9, 12, (3, 3), stride=2)
        self.conv4 = torch.nn.ConvTranspose2d(15, 24, (4, 4), stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 41, 41)

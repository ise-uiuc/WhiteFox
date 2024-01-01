
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 2)
        self.conv1 = torch.nn.ConvTranspose2d(32, 64, (3, 3), 2)
        self.conv2 = torch.nn.ConvTranspose2d(64, 128, kernel_size=(2,2),stride=(3,3))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v1)
        v4 = self.conv2(v3)
        v4 = torch.sigmoid(v4)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 5, 5)

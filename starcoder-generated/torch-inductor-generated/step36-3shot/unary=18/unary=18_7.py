
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=9, stride=2, padding=4)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 31, 15)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 128, kernel_size=11, padding=5, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=3)
        self.conv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=7, padding=5, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 1, kernel_size=1, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

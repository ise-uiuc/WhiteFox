
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = F.tanh(self.conv4(x))
        return x
# Inputs to the model
x1 = torch.randn(1, 128, 2, 2)

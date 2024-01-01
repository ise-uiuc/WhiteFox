
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.conv1 = nn.ConvTranspose2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

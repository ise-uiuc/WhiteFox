
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(192, 96, 5, stride=1, padding=4)
        self.conv2 = torch.nn.ConvTranspose2d(96, 96, 3, stride=2, padding=2, output_padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(96, 64, 4, stride=2, padding=2, output_padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x = torch.randn(2, 192, 1, 1)

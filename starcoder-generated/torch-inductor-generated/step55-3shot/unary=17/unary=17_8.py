
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 8, 3, padding=1, stride=2, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 16, 3, padding=1, stride=2, output_padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2, output_padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2, output_padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.relu(v1)
        v2 = self.conv2(v1)
        v2 = torch.relu(v2)
        v3 = self.conv3(v2)
        v3 = torch.relu(v3)
        v4 = self.conv4(v3)
        v4 = torch.relu(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)

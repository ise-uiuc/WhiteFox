
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(4, 1, 3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(4, 4, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        v6 = self.max_pool(v5)
        return v6

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=1 * 1 * 3, out_features=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.flatten(v2)
        v4 = self.fc1(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

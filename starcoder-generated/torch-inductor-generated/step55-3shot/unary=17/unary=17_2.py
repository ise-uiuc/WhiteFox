
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(20, 20, 3, padding=1, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(20, 20, 3, padding=1, stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(20, 1, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 20, 128, 128)

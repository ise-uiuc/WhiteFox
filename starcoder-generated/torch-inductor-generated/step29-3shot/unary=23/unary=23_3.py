
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.ConvTranspose2d(3, 5, 3)
        self.conv2 = torch.nn.ConvTranspose2d(5, 3, 2)
        self.conv3 = torch.nn.ConvTranspose2d(3, 5, 3)
        self.conv4 = torch.nn.ConvTranspose2d(5, 1, 2)
    def forward(self, x1):
        v1 = self.relu1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v1)
        v5 = self.conv4(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)

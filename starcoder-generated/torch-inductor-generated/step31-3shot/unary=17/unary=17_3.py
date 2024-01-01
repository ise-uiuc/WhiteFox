
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 20, 3, padding=1, stride=2, output_padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(20, 10, 3, padding=1, stride=2, output_padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(10, 5, 3, padding=1, stride=2, output_padding=0)
        self.conv4 = torch.nn.ConvTranspose2d(5, 1, 3, padding=1, stride=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)

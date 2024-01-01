
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(30, 50, 4, stride=2, padding=1, output_padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(50, 25, 4, stride=2, padding=1, output_padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(25, 1, 4, stride=2, padding=1, output_padding=0)
    def forward(self, x1):
        v0 = F.relu(x1)
        v1 = self.conv2(v0)
        v2 = F.relu(v1)
        v3 = self.conv1(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 30, 28, 28)

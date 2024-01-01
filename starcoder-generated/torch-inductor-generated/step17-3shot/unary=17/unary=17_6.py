
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True)
        self.conv3 = torch.nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)

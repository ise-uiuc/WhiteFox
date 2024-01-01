
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(10, 200, 3, padding=0, stride=1)
        self.conv_2 = torch.nn.ConvTranspose2d(200, 100, 3, padding=0, stride=1)
        self.conv_3 = torch.nn.Conv2d(100, 256, 3, padding=2, stride=2)
        self.conv_4 = torch.nn.ConvTranspose2d(256, 100, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_2(v2)
        v4 = self.conv_3(v3)
        v5 = torch.relu(v4)
        v6 = self.conv_4(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)

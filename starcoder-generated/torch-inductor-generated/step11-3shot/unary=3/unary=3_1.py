
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(8, 4, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = self.conv5(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 8, 112, 112)

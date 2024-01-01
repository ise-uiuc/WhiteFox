
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(256, 3, 5, stride=3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 - 1.5
        v3 = F.elu(v2, 0.5)
        v4 = v3 - 0.001
        v5 = self.conv2(v3)
        v6 = F.elu(v4, 1.5)
        v7 = v5 - 0.002
        v8 = self.conv3(v6)
        v9 = F.elu(v7, 2.0)
        v10 = v8 - 0.003
        v11 = self.conv4(v9)
        return v11 
# Inputs to the model
x = torch.randn(1, 1, 13, 13)

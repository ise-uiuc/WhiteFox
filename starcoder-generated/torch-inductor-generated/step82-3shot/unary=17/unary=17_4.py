
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 32, 3, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(16, 1, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_2(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

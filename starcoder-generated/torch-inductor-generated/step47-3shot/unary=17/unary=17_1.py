
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=0)
    def forward(self, x0, x1):
        v0 = self.conv(x0)
        v1 = self.conv1(x1)
        v2 = self.conv(v0)
        v3 = self.conv1(v1)
        v4 = self.conv(v2)
        v5 = self.conv1(v3)
        v6 = torch.cat([v4, v5], dim=1)
        v7 = torch.split(v6, [1, 1], dim=1)
        v8 = v7[0]
        v9 = v7[1]
        v10 = torch.relu(v9)
        v11 = torch.sigmoid(v9)
        v12 = torch.tanh(v8)
        v13 = v12 + v10 - v11
        v14 = torch.squeeze(v13)
        v15 = v14 + v9 - v11
        v16 = self.conv(v15)
        v17 = torch.max(v16)
        v18 = torch.argmax(v17)
        return v18
# Inputs to the model
x0 = torch.randn(1, 1, 32, 32)
x1 = torch.randn(1, 1, 64, 64)

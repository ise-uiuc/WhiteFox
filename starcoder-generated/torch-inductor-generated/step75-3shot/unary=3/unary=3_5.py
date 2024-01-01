
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 7, stride=2, padding=3)
        self.dropout = torch.nn.Dropout2d(0.083588)
        self.conv2 = torch.nn.Conv2d(18, 18, 5, stride=1, padding=2)
        self.dropout2 = torch.nn.Dropout2d(0.190601)
        self.conv3 = torch.nn.ConvTranspose2d(18, 1, 1, stride=2, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, 1)
        v3 = torch.reshape(v2, (-1, 18, 1, 1))
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = v8 + x2
        v10 = self.dropout(v9)
        v11 = self.conv2(v10)
        v12 = torch.flatten(v11, 1)
        v13 = torch.reshape(v12, (-1, 18, 1, 1))
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = v18 + x2
        v20 = self.dropout2(v19)
        v21 = self.conv3(v20)
        return v21
# Inputs to the model
x1 = torch.randn(1, 1, 156, 156)
x2 = torch.randn(1, 18, 156, 156)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v7 = self.conv(x1)
        v8 = self.conv(x1)
        v9 = self.conv(x1)
        v10 = self.conv(x1)
        v11 = self.conv(x1)
        v12 = self.conv(x1)
        v13 = self.conv(x1)
        v14 = self.conv(x1)
        v15 = self.conv(x1)
        v16 = self.conv(x1)
        v17 = self.conv(x1)
        v18 = self.conv(x1)
        v19 = self.conv(x1)
        v20 = self.conv(x1)
        v21 = self.conv(x1)
        v22 = self.conv(x1)
        v23 = self.conv(x1)
        v24 = self.conv(x1)
        v25 = self.conv(x1)
        v26 = self.conv(x1)
        v27 = self.conv(x1)
        v28 = self.conv(x1)
        v29 = self.conv(x1)
        v30 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + \
              v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 + v20 + \
              v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28 + v29 + v30
        v31 = torch.relu(v30)
        return v31
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

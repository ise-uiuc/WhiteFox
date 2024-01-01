
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 10, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.relu6(v1)
        v3 = self.relu6(v2)
        v4 = self.conv_transpose(v1)
        v5 = v4 + 3
        v6 = self.relu6(v5)
        v7 = self.relu6(v6)
        v8 = self.relu6(v7)
        v9 = self.conv_transpose(v2)
        v10 = v9 + 3
        v11 = self.relu6(v10)
        v12 = self.relu6(v11)
        v13 = self.relu6(v12)
        v14 = torch.min(v8, v13)
        v15 = torch.max(v14, v15)
        v16 = v15 / 6
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

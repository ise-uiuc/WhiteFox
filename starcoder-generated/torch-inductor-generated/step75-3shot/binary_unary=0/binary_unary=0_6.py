
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv1(v2)
        v4 = self.conv1(v3)
        v5 = self.conv1(v4)
        v6 = self.conv1(v5)
        v7 = self.conv1(v6)
        v8 = self.conv1(v7)
        v9 = self.conv1(v8)
        v10 = self.conv1(v9)
        v11 = self.conv1(v10)
        #v11 = v9 + v3
        v12 = self.conv1(v11)
        v13 = self.conv1(v12)
        v14 = self.conv1(v13)
        v15 = self.conv1(v14)
        v16 = self.conv1(v15)
        v17 = self.conv1(v16)
        v18 = self.conv1(v17)
        v19 = self.conv1(v18)
        v20 = self.conv1(v19)
        v21 = self.conv1(v20)
        #v19 = v13 + self.conv1(v20)
        #v21 = v21 + self.conv1(v9)
        v22 = torch.conv2d(v21, weight=torch.rand(3, 3))
        return v22
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)

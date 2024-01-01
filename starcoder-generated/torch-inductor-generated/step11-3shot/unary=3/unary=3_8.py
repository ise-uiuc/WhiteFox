
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 2, 3, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.cat([v1,v2],axis=1)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv3(v8)
        v10 = self.conv4(v9)
        v11 = torch.cat([v8,v10],axis=1)
        v12 = v11 * 0.5
        v13 = v11 * 0.7071067811865476
        v14 = torch.erf(v13)
        v15 = v14 + 1
        v16 = v12 * v15
        v17 = self.conv5(v16)
        v18 = self.conv6(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 8, 111, 111)
x2 = torch.randn(1, 8, 112, 112)

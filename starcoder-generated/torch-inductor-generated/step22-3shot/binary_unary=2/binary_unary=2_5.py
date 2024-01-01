
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(4, 1, 2, stride=1)
    def forward(self, x1):
        v1 = torch.transpose(x1, 2, 3)
        v2 = x1.size()
        v3 = v2[2]
        v4 = x1.size()
        v5 = v4[3]
        v6 = torch.add(v3, -v5)
        v7 = torch.div(v6, 2)
        v8 = v2[3]
        v9 = torch.div(v8, 2)
        v10 = torch.zeros(1, 1, v7, v9, dtype=torch.float)
        v11 = torch.add(v1, v10)
        v12 = self.conv2(v10)
        v13 = torch.transpose(v11, 2, 3)
        v14 = torch.squeeze(v12, 0)
        v15 = torch.squeeze(v13, 0)
        v16 = torch.add(v15, v14)
        return v16
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)

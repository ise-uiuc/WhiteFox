
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.ParameterDict({
            'w1': torch.nn.Parameter(torch.randn(54, 473, 3)),
            'w2': torch.nn.Parameter(torch.randn(3, 1))
        })
        self.dropout = torch.nn.Dropout(p=0.1)
        self.conv = torch.nn.Conv2d(3, 30, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(30, 25, 5, stride=2, padding=0)
        self.conv3 = torch.nn.Conv1d(840, 29, 10, stride=14, padding=0)
        self.conv4 = torch.nn.Conv1d(29, 30, 20, stride=17, padding=0)
        self.conv5 = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv3(v12)
        v14 = self.conv4(v12)
        v15 = torch.flatten(v14, 1)
        v16 = self.dropout(v15)
        v17 = v16.matmul(self.weight['w1'])
        v18 = v17.transpose(0, 1)
        v19 = self.conv5(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 3, 199, 397)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=2, padding=(2, 3))
        self.conv2 = torch.nn.Conv2d(64, 83, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(83, 22, 6, stride=6, padding=2)
        self.conv4 = torch.nn.Conv2d(22, 11, 3, stride=2, padding=0)
    def forward(self, x52):
        v1 = self.conv1(x52)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        v12 = self.conv3(v11)
        v13 = self.conv4(v12)
        v14 = v13 * 0.5
        v15 = v13 * v13
        v16 = v15 * v13
        v17 = v16 * 0.044715
        v18 = v13 + v17
        v19 = v18 * 0.7978845608028654
        v20 = torch.tanh(v19)
        v21 = v20 + 1
        v22 = v14 * v21
        return v22
# Inputs to the model
x52 = torch.randn(1, 3, 32, 74)

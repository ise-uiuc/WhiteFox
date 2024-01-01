
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(9, 8, 5, stride=2, padding=3)
        self.conv2 = torch.nn.Conv1d(12, 17, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(17, 15, 3, stride=2, padding=2)
    def forward(self, x17):
        v1 = self.conv(x17)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = v10 * 0.32710867602342303
        v12 = self.conv2(x17)
        v13 = v12 + v1
        v14 = v13 - 0.41612689379109553
        v15 = torch.sigmoid(v14)
        v16 = v11 * v15
        v17 = v16 + 1
        v18 = self.conv3(v17)
        v19 = v18 * 0.33548051230267443
        v20 = v19 * 0.5582855661136502
        v21 = v19 * v19
        v22 = v21 * v19
        v23 = v22 * 0.0025896553010033047
        v24 = v20 + v23
        v25 = v24 * 0.5183906180303593
        v26 = torch.log(v25)
        v27 = v16 * v26
        return v27
# Inputs to the model
x17 = torch.randn(1, 9, 61)

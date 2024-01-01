
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(26, 24, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = torch.relu(v1)
        v5 = torch.relu(v2)
        v6 = torch.relu(v3)
        v7 = torch.cat([v4,v5,v6], axis=1)
        v8 = self.conv1(v7)
        v9 = torch.relu(v8)
        v10 = self.conv1(v9)
        v11 = torch.cat([(v10 + v1),v10,v9,v8,v7,v6,v5,v4], axis=1)
        v12 = torch.transpose(v11, 0, 1)
        v13 = torch.matmul(v12, v11)
        v14 = torch.transpose(v13, 0, 1)
        v15 = torch.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 26, 64, 64)

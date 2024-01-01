
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = x1.mean(dim=0, keepdim=True)
        v2 = x1.mean(dim=2, keepdim=False)
        v3 = x1.mean(dim=3, keepdim=True)
        v4 = x2.flatten(start_dim=2, end_dim=-1)
        v5 = x3.flatten(end_dim=2)
        v6 = self.conv1(x1)
        v7 = self.conv2(v1)
        v8 = self.conv3(v2)
        v9 = v7 + v8
        v10 = torch.relu(v9)
        v11 = self.conv1(x2)
        v12 = self.conv2(v4)
        v13 = self.conv3(v3)
        v14 = v12 + v13
        v15 = torch.relu(v14)
        v16 = self.conv1(x3)
        v17 = self.conv2(v5)
        v18 = self.conv3(v16)
        v19 = v17 + v18
        v20 = torch.relu(v19)
        return v20
# Inputs to the model
x1 = torch.randn(16, 1, 128, 128)
x2 = torch.randn(16, 3, 16, 16)
x3 = torch.randn(16, 1, 32, 32)

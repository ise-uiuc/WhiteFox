
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 6, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 6, 4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.norm(v7, dim=-1)
        v9 = torch.transpose(v8, -1, 1)
        v10 = torch.matmul(v9, v7)
        v11 = torch.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 4, 31, 55)

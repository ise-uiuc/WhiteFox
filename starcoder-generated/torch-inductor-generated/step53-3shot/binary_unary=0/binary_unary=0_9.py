
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, v, x0, x1):
        q1 = v[0].cuda()
        q2 = v[1][0].cuda()
        v1 = self.conv1(q1)
        v2 = v1 + q2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        a1 = self.conv2(q2)
        v5 = v4 + a1
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + v[2].cuda()
        v9 = torch.relu(v8)
        v10 = self.conv1(v9)
        v11 = v10 + q1
        v12 = torch.relu(v11)
        # output = torch.stack((v6, v2, v5, v9))
        return v12
# Inputs to the model
v = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64)]
x = torch.randn(1, 16, 64, 64)

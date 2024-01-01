
class Model(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv1 = torch.nn.Conv2d(self.num_features, self.num_features, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(self.num_features, self.num_features, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(self.num_features, self.num_features, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(self.num_features, self.num_features, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
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
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv4(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 65, 384, 765)

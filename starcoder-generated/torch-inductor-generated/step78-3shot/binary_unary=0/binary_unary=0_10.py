
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_23 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_45 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_67 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_89 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_101 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 + v1
        v_103 = torch.relu(v3)
        v4 = self.conv_23(v_103)
        v5 = v4 + v1
        v_205 = torch.relu(v5)
        v6 = self.conv_45(v_205)
        v7 = v6 + v3
        v_307 = torch.relu(v7)
        v8 = self.conv_67(v_307)
        v9 = v8 + v5
        v_409 = torch.relu(v9)
        v10 = self.conv_89(v_409)
        v11 = v10 + v7
        v_511 = torch.relu(v11)
        v12 = self.conv_101(v_511)
        v13 = v12 + v_103
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x = torch.randn(1, 16, 64, 64)

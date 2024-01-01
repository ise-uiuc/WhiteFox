
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = torch.nn.Conv2d(3, 12, 1, stride=1, padding=0)
        self.conv_1_2 = torch.nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.conv_2_1 = torch.nn.Conv2d(12, 12, 1, stride=1, padding=0)
        self.conv_2_2 = torch.nn.Conv2d(12, 12, 1, stride=1, padding=0)
        self.conv_2_3 = torch.nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.conv_2_4 = torch.nn.Conv2d(12, 12, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_1_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_1_2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_2_1(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_2_2(v6)
        v8 = torch.relu(v7)
        v9 = self.conv_2_3(v8)
        v10 = torch.relu(v9)
        v11 = self.conv_2_4(v10)
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)

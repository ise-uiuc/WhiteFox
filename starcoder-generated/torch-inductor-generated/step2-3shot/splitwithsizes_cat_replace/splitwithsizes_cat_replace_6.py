
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split_sizes_2_3 = 2
        self.split_sizes_5_8 = 5
        self.conv1_1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv1_2 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv2_1 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv1_1(x1)
        v2 = self.conv1_2(x2)
        v3 = torch.split(v1, [self.split_sizes_2_3, self.split_sizes_5_8], 1)
        v4 = torch.split(v2, [self.split_sizes_2_3, self.split_sizes_5_8], 1)
        v5 = torch.cat([v3[0], v4[0]], 1)
        v6 = torch.cat([v3[1], v4[1]], 1)
        v7 = self.conv2_1(v5)
        v8 = self.conv2_2(v6)
        v9 = torch.cat([v7, v8], 0)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

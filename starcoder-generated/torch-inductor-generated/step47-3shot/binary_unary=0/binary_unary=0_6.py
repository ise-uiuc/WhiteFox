
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v6 = v1 + v1
        v7 = torch.relu(v6)
        v8 = v7 + v1
        v9 = torch.relu(v8)
        v10 = self.conv2(v9)
        v13 = v10 + v10
        v14 = torch.relu(v13)
        v15 = v14 + v10
        v16 = torch.relu(v15)
        v17 = self.conv3(v16)
        v19 = torch.nn.functional.pixel_shuffle(v17, 2)
        v20 = v19 + x2
        v21 = torch.relu(v20)
        return v21
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)

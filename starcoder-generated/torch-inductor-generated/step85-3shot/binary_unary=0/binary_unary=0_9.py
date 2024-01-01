
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, groups=2)
        self.conv3 = torch.nn.ConvTranspose2d(16, 16, 7, stride=2, padding=3, output_padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=8)
        self.conv5 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, groups=8)
        self.conv6 = torch.nn.Conv2dTranspose2d(16, 16, 7, stride=2, padding=3, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x1
        v6 = torch.nn.functional.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + x1
        v9 = torch.nn.functional.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 + x1
        v12 = torch.nn.functional.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 + x1
        v15 = torch.nn.functional.relu(v14)
        v16 = self.conv6(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.add = torch.add
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = v1 + y # Not necessarily for fusion. But may be.
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        m1 = torch.nn.functional.interpolate(v4, scale_factor=1.0, mode='bicubic', align_corners=True)
        v5 = m1 + v4
        v6 = torch.relu(v5)
        m2 = torch.nn.functional.interpolate(v6, scale_factor=1.0, mode='bicubic', align_corners=True)
        v7 = m2 + v3
        v8 = self.conv3(v7)
        v9 = v8 + v2
        v10 = torch.relu(v9)
        m3 = torch.nn.functional.interpolate(v10, scale_factor=1.0, mode='bicubic', align_corners=True)
        v11 = m3 + v4
        v12 = torch.relu(v11)
        m4 = torch.nn.functional.interpolate(v12, scale_factor=1.0, mode='bicubic', align_corners=True)
        v13 = m4 + v7
        v14 = self.conv4(v13)
        m5 = torch.nn.functional.max_pool2d(v14, kernel_size=4)
        v15 = m5 + v10
        v16 = torch.relu(v15)
        m6 = torch.nn.functional.max_pool2d(v16, kernel_size=4)
        v17 = m6 + v9
        v18 = torch.relu(v17)
        m7 = torch.nn.functional.max_pool2d(v18, kernel_size=4)
        v19 = m7 + v13
        v20 = self.conv5(v19)
        v21 = torch.reshape(v20, (-1, 1))
        v22 = v21 + v10
        v23 = torch.relu(v22)
        m8 = torch.nn.functional.max_pool2d(v23, kernel_size=4)
        v24 = m8 + v21
        v25 = torch.relu(v24)
        return v25
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)

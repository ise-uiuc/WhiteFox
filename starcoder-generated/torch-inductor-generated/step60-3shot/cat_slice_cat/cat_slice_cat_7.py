
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32)
        self.conv = torch.nn.Conv2d(1, 64, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(64, 1, 1, stride=1, padding=0)
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.fc(x1)
        v2 = v1.flatten(start_dim=1)
        v3 = v2.matmul(x2)
        v4 = v3.reshape(-1, 1, 8, 24)
        v5 = self.conv8(v4)
        v6 = v5 + x3
        v7 = v6 + x4
        v8 = self.conv9(v7)
        v9 = v8.flatten(start_dim=1)
        v10 = v9 * x5
        v11 = v10.matmul(x6)
        v12 = v11.reshape(-1, 1, 8, 8)
        v13 = self.conv10(v12)
        v14 = v13 + x7
        v15 = v14 + x8
        v16 = self.conv11(v15)
        v17 = v16.flatten(start_dim=1)
        return v17

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(32, 8)
x3 = torch.randn(1, 128, 4, 8)
x4 = torch.randn(1, 128, 4, 8)
x5 = torch.randn(32)
x6 = torch.randn(8, 16)
x7 = torch.randn(1, 128, 4, 8)
x8 = torch.randn(1, 128, 4, 8)
